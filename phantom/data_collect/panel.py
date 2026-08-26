"""collect web control panel (requirements 3 + 4).

Same architecture as the hardware-proven phantom panel (stdlib-only
ThreadingHTTPServer, SSE state push at 2 Hz, buttons -> thread-safe queue
merged by the session loop; rerun does ALL image/plot streaming and is
embedded as an iframe) with the collect feature set:

  - SAFEGUARD card: red latched banner on a DM-Tac force trip, an explicit
    RESUME button (the only way to continue — no restart needed), an
    enable/disable toggle and live-editable force / depth thresholds
    (POST /api/safeguard applies immediately);
  - session wizard with the collection MODE switch (full | lite);
  - continuous-gripper live bar + leader-tick Calibrate button;
  - offload card: external-drive status, session-end progress, manual retry;
  - episodes card: browse episodes already on the external drive (or still
    in staging), run the recording VERIFIER (per-stream report table) and
    PLAY any episode back into the embedded rerun viewer — all modalities,
    with pause/stop/speed (playback.py).

Trusted-LAN only (buttons move a robot); binds 127.0.0.1 unless configured.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import types
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

BUTTONS = ("start_stop", "success", "fail", "abort", "resume_safeguard",
           "calibrate_gripper", "reengage", "quit")
PHASES = ("setup", "starting", "running", "stopping")
_TASK_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


@dataclass
class CollectPanelState:
    phase: str = "setup"
    error: str = ""
    # fine-grained bring-up / teardown step (spinner sub-line on the panel);
    # "" when idle or running, kept as the last step reached on an error
    busy_detail: str = ""
    task: str = ""
    operator: str = ""
    mode: str = "full"
    recording: bool = False
    # episode contour state machine (drives the Episode buttons + spinner):
    # idle -> recording -> finalizing -> awaiting_verdict -> saving -> idle.
    # episode_detail is the live English status line under the block.
    episode_phase: str = "idle"
    episode_detail: str = ""
    episode: str = ""
    ep_count: int = 0
    target_episodes: int = 0
    episodes: list = field(default_factory=list)
    staging_dir: str = ""
    # safeguard
    safeguard_enabled: bool = True
    safeguard_force_limit_n: float = 4.0
    safeguard_depth_limit: float = 0.5
    safeguard_tripped: bool = False
    safeguard_msg: str = ""
    # holds / live values
    arm_hold: str = ""
    # True while reengage() is blocking the session loop (up to ~10 s:
    # control teardown + settle + up to 3 script-start attempts)
    reengaging: bool = False
    workspace_hold: bool = False
    track_phase: str = ""
    tick_hz: float = 0.0
    wrench_N: float = 0.0
    tactile_force_N: float = 0.0
    gripper: float = 0.0
    calibrating: bool = False
    calib_msg: str = ""
    # offload
    offload_running: bool = False
    offload_ok: bool = True
    offload_done: int = 0
    offload_total: int = 0
    offload_msg: str = ""
    # playback / verify (episodes already on the drive or in staging)
    playback_episode: str = ""
    playback_state: str = "idle"
    playback_pos_s: float = 0.0
    playback_dur_s: float = 0.0
    playback_speed: float = 1.0
    playback_msg: str = ""
    verify_running: bool = False
    verify_report: dict | None = None
    rerun_url: str = ""
    viewer_native: bool = False    # native wgpu viewer in use -> hide the iframe
    # configured viewer mode (from cc.rerun.viewer, seeded at startup). "native"
    # -> the panel is a pure control surface (no embedded viewer block at all;
    # the rerun window lives on the rig display). "web" -> the old embedded
    # iframe path, kept as a working fallback.
    viewer_mode: str = "web"
    t_ep_start: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def add_episode(self, *, name: str, outcome: str, tags=()) -> None:
        with self._lock:
            self.episodes.append({"name": name, "outcome": outcome,
                                  "tags": list(tags), "t": time.time()})

    def reset_session(self, **kw) -> None:
        with self._lock:
            self.episodes = []
            self.ep_count = 0
            self.recording = False
            self.episode_phase = "idle"
            self.episode_detail = ""
            self.safeguard_tripped = False
            self.safeguard_msg = ""
            self.arm_hold = ""
            self.reengaging = False
            self.episode = ""
            self.error = ""
            self.busy_detail = ""
            self.rerun_url = ""
            self.viewer_native = False
            self.offload_msg = ""
            self.playback_state = "idle"
            self.playback_episode = ""
            self.playback_msg = ""
            self.playback_pos_s = 0.0
            self.playback_dur_s = 0.0
            self.verify_report = None
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            d = {k: getattr(self, k) for k in self.__dataclass_fields__
                 if not k.startswith("_")}
            d["episodes"] = list(d["episodes"])
        d["ep_seconds"] = round(time.time() - d.pop("t_ep_start"), 1) \
            if d["recording"] else 0.0
        return d


class CollectRunner:
    """Bridges the panel's session/offload endpoints to CollectApp."""

    def __init__(self, app, panel):
        self.app = app
        self.panel = panel
        self.playback = None       # PlaybackController, attached by collect.py
        self._thread: threading.Thread | None = None
        self._offload_thread: threading.Thread | None = None
        # ThreadingHTTPServer handles each POST on its own thread — without
        # this lock two concurrent /api/session/start pass the is_running
        # check together and BOTH open the (single-open!) hardware
        self._start_lock = threading.Lock()

    @property
    def gate_lock(self) -> threading.Lock:
        """Shared start gate: PlaybackController serializes its own start
        checks on this same lock, so 'session vs playback' mutual exclusion
        cannot race between two HTTP threads."""
        return self._start_lock

    def start_session(self, cfg: dict) -> str | None:
        with self._start_lock:
            if self.is_running():
                return "a session is already running"
            if self._offload_thread is not None and self._offload_thread.is_alive():
                return "offload in progress — wait for it to finish"
            if self.playback is not None and self.playback.blocking():
                return "episode playback/verify in progress — stop it first"
            task = str(cfg.get("task", "")).strip()
            if not _TASK_RE.match(task):
                return "task name: letters/digits/_- (1-64 chars)"
            mode = cfg.get("mode", self.app.cc.default_mode)
            if mode not in ("full", "lite"):
                return f"unknown mode {mode!r}"
            s = types.SimpleNamespace(
                task=task,
                text=str(cfg.get("text", "")).strip(),
                operator=str(cfg.get("operator", "")).strip(),
                mode=mode,
                target_episodes=max(0, int(cfg.get("target_episodes", 0) or 0)),
            )
            self.panel.state.reset_session(
                phase="starting", task=task, operator=s.operator, mode=mode,
                target_episodes=s.target_episodes)
            # Nothing queued before this moment belongs to this session. Any
            # leftover would be popped on the FIRST loop tick: a stale
            # 'start_stop' auto-records a phantom episode during the engage
            # glide, a stale 'quit' ends the freshly brought-up session.
            # push_button already drops commands while no session runs; this
            # covers whatever slipped in during bring-up of a PREVIOUS one.
            self.panel.pop_buttons()
            self._thread = threading.Thread(target=self._run, args=(s,),
                                            daemon=True, name="collect-session")
            self._thread.start()
            return None

    def stop_session(self) -> None:
        self.panel.state.update(phase="stopping")
        self.panel.push_button("quit")

    def start_offload(self) -> str | None:
        with self._start_lock:
            if self.is_running():
                return ("stop the session first — offload runs automatically "
                        "at session end")
            if self._offload_thread is not None and self._offload_thread.is_alive():
                return "offload already running"
            if self.playback is not None and self.playback.blocking():
                # the offload rewrites drive session dirs and deletes staging
                # copies — never while an episode is being read back
                return "episode playback/verify in progress — stop it first"
            self._offload_thread = threading.Thread(
                target=self.app.offload, daemon=True, name="collect-offload")
            self._offload_thread.start()
            return None

    def busy(self) -> bool:
        """A session or an offload is in flight (graceful-shutdown gate)."""
        return (self.is_running()
                or (self._offload_thread is not None
                    and self._offload_thread.is_alive()))

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self, s) -> None:
        # run_session owns the phase now: it stays "starting" through bring-up
        # (updating busy_detail per step), flips to "running" when the main
        # loop goes live, and to "stopping" during teardown/offload.
        try:
            self.app.run_session(s)
            self.panel.state.update(phase="setup", busy_detail="")
        except Exception as e:
            log.exception("session failed")
            # keep busy_detail = the last bring-up step so the operator sees
            # where it died; the error banner shows the exception
            self.panel.state.update(phase="setup", error=str(e))
        finally:
            # a session ending (clean OR via a raised fault) always clears the
            # episode contour, so the Episode buttons can't be left stuck in
            # recording/awaiting_verdict once no session is running
            self.panel.state.update(episode_phase="idle", episode_detail="")
            self.panel.pop_buttons()


class CollectPanel:
    def __init__(self, state: CollectPanelState, *, host: str = "127.0.0.1",
                 port: int = 8899, runner: CollectRunner | None = None,
                 setup_provider=None, safeguard_provider=None, playback=None):
        """safeguard_provider: () -> TactileSafeguard | None — the live
        safeguard of the current session (POST /api/safeguard target).
        playback: PlaybackController — GET /api/episodes + POST
        /api/playback/* (attached by scripts/collect.py)."""
        self.state = state
        self.host = host
        self.port = port
        self.runner = runner
        self.setup_provider = setup_provider
        self.safeguard_provider = safeguard_provider
        self.playback = playback
        self._cmds: queue.Queue[str] = queue.Queue()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def pop_buttons(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        while True:
            try:
                out[self._cmds.get_nowait()] = True
            except queue.Empty:
                return out

    def push_button(self, name: str) -> bool:
        """Queue one command for the session loop. Returns False if it was
        dropped because no session is running.

        The queue has exactly one consumer — the session loop — so a button
        pressed while idle (or during the ~30 s bring-up, when the Episode
        card still offers Start) is not "queued for later", it is queued for
        the NEXT session and fires on its first tick: a phantom episode that
        nobody started, or an instant quit. Dropping it at the door is the
        only place that knows the button is homeless."""
        if self.runner is not None and not self.runner.is_running():
            log.info("panel button %r ignored: no session is running", name)
            return False
        self._cmds.put(name)
        return True

    # ------------------------------------------------------------------
    def start(self) -> None:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                n = int(self.headers.get("Content-Length", 0))
                try:
                    obj = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return {}
                # a valid-JSON non-object body ([1], "x", 5) must not crash
                # the handler thread on .get()
                return obj if isinstance(obj, dict) else {}

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/status":
                    self._json(panel.state.snapshot())
                elif self.path == "/api/setup":
                    self._json(panel.setup_provider() if panel.setup_provider else {})
                elif self.path == "/api/episodes":
                    if panel.playback is None:
                        self._json({"error": "playback not available"}, 409)
                        return
                    try:
                        self._json({"episodes": panel.playback.scan()})
                    except Exception as e:
                        self._json({"error": f"episode scan failed: {e}"}, 500)
                elif self.path == "/api/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
                            data = json.dumps(panel.state.snapshot())
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(0.5)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                if self.path == "/api/cmd":
                    btn = self._body().get("button")
                    if btn in BUTTONS:
                        # push_button drops it when no session is running
                        queued = panel.push_button(btn)
                        self._json({"ok": True, "button": btn,
                                    "queued": queued})
                    else:
                        self._json({"error": f"unknown button {btn!r}"}, 400)
                elif self.path == "/api/safeguard":
                    sg = (panel.safeguard_provider()
                          if panel.safeguard_provider else None)
                    body = self._body()
                    kw = {}
                    if "enabled" in body:
                        kw["enabled"] = bool(body["enabled"])
                    for key in ("force_limit_n", "depth_limit"):
                        if key in body:
                            try:
                                kw[key] = float(body[key])
                            except (TypeError, ValueError):
                                self._json({"error": f"{key} must be a number"}, 400)
                                return
                    if sg is None:
                        # no live session: remember the values in panel state so
                        # the next session picks them up via the wizard display
                        panel.state.update(**{
                            "safeguard_enabled": kw.get(
                                "enabled", panel.state.safeguard_enabled),
                            "safeguard_force_limit_n": kw.get(
                                "force_limit_n",
                                panel.state.safeguard_force_limit_n),
                            "safeguard_depth_limit": kw.get(
                                "depth_limit", panel.state.safeguard_depth_limit)})
                        self._json({"ok": True, "live": False})
                        return
                    try:
                        sg.configure(**kw)
                    except ValueError as e:
                        self._json({"error": str(e)}, 400)
                        return
                    panel.state.update(safeguard_enabled=sg.enabled,
                                       safeguard_force_limit_n=sg.force_limit_n,
                                       safeguard_depth_limit=sg.depth_limit)
                    self._json({"ok": True, "live": True})
                elif self.path == "/api/session/start":
                    if panel.runner is None:
                        self._json({"error": "no runner attached"}, 409)
                        return
                    err = panel.runner.start_session(self._body())
                    self._json({"ok": True} if err is None else {"error": err},
                               200 if err is None else 409)
                elif self.path == "/api/session/stop":
                    if panel.runner is None:
                        self._json({"error": "no runner attached"}, 409)
                    else:
                        panel.runner.stop_session()
                        self._json({"ok": True})
                elif self.path == "/api/offload":
                    if panel.runner is None:
                        self._json({"error": "no runner attached"}, 409)
                    else:
                        err = panel.runner.start_offload()
                        self._json({"ok": True} if err is None else {"error": err},
                                   200 if err is None else 409)
                elif self.path.startswith("/api/playback/"):
                    pb = panel.playback
                    if pb is None:
                        self._json({"error": "playback not available"}, 409)
                        return
                    op = self.path.rsplit("/", 1)[1]
                    body = self._body()
                    err = None
                    if op in ("play", "speed"):
                        try:
                            speed = float(body.get("speed", 1.0))
                        except (TypeError, ValueError):
                            self._json({"error": "speed must be a number"}, 400)
                            return
                        if op == "play":
                            err = pb.play(str(body.get("path", "")), speed=speed)
                        else:
                            pb.set_speed(speed)
                    elif op == "verify":
                        err = pb.verify(str(body.get("path", "")))
                    elif op == "pause":
                        pb.pause()
                    elif op == "resume":
                        pb.resume()
                    elif op == "stop":
                        pb.stop()
                    else:
                        self._json({"error": "not found"}, 404)
                        return
                    self._json({"ok": True} if err is None else {"error": err},
                               200 if err is None else 409)
                else:
                    self._json({"error": "not found"}, 404)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="collect-panel")
        self._thread.start()
        log.info("collect panel at %s", self.url)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


# ---------------------------------------------------------------------------
# the page — self-contained (no CDNs; works offline on the rig)
# ---------------------------------------------------------------------------

_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>collect — PHANTOM teleop + data collection</title>
<style>
:root{--bg:#111418;--card:#1a1f26;--ink:#e6e9ef;--dim:#8a93a3;--line:#2a3140;
--ok:#3fb68b;--warn:#e0a93e;--bad:#e05d5d;--acc:#4d9fec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1500px;margin:0 auto;padding:14px;display:grid;gap:12px;
grid-template-columns:360px minmax(0,1fr);align-items:start}
.controls{display:flex;flex-direction:column;gap:12px;min-width:0}
.viewer{min-width:0;display:flex;flex-direction:column;gap:12px}
.statusband{min-width:0}
/* the control grid is a single stacked column on desktop (the left rail);
   portrait turns .grid3 into three dense columns and .colC into a stack */
.grid3{display:flex;flex-direction:column;gap:12px;min-width:0}
.colC{display:contents}
h1{font-size:16px;margin:0 0 4px}h2{font-size:13px;margin:0 0 8px;color:var(--dim);
text-transform:uppercase;letter-spacing:.06em}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px}
.statusband h1{white-space:nowrap}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;
font-weight:600;background:#26303e;color:var(--dim);margin-right:6px}
.pill.on{background:#173325;color:var(--ok)}
.pill.rec{background:#3a1520;color:#ff7d9c}
.pill.bad{background:#3a1a1a;color:var(--bad)}
.pill.warn{background:#38300f;color:var(--warn)}
button{background:#26303e;color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:8px 12px;cursor:pointer;font-weight:600}
button:hover{background:#2f3b4d}button:disabled{opacity:.4;cursor:default}
button.acc{background:var(--acc);border-color:var(--acc);color:#08131f}
button.bad{background:var(--bad);border-color:var(--bad);color:#1f0808}
button.big{font-size:15px;padding:12px 16px;width:100%}
/* session start/stop busy state: disabled button with an inline spinner and,
   directly under it, a live one-line step read-out (busy_detail). The detail
   line reserves its height at all times so nothing jumps as steps change. */
button.busy{display:flex;align-items:center;justify-content:center;gap:8px}
.spinner{width:15px;height:15px;flex:0 0 auto;border-radius:50%;
border:2px solid rgba(230,233,239,.28);border-top-color:var(--ink);
animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#busyDetail,#epDetail{min-height:15px;margin-top:5px;font-size:11px;
color:var(--dim);text-align:center;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
/* SSE dropped (server restarting): freeze all controls so clicks can't fire
   into a dead socket, and show the reconnecting pill in the status ribbon */
body.disconnected button{opacity:.4;pointer-events:none}
#connPill{margin-left:2px}
input,select{background:#0e1116;color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:7px 9px;width:100%}
label{font-size:12px;color:var(--dim);display:block;margin:8px 0 3px}
.row{display:flex;gap:8px}.row>*{flex:1}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:2px 0}
.kv b{font-variant-numeric:tabular-nums}
.banner{display:none;background:#3a1a1a;border:1px solid var(--bad);
border-radius:10px;padding:12px;margin-bottom:12px}
.banner.show{display:block}
.banner h3{margin:0 0 6px;color:var(--bad);font-size:15px}
.bar{height:10px;background:#0e1116;border-radius:6px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);width:0%}
.eplist{max-height:180px;overflow-y:auto;font-size:12px}
.eplist div{padding:3px 0;border-bottom:1px solid var(--line);color:var(--dim)}
.eplist .success{color:var(--ok)}.eplist .fail,.eplist .safeguard{color:var(--bad)}
.eplist .unjudged{color:var(--warn)}
iframe{width:100%;height:640px;border:1px solid var(--line);border-radius:10px;
background:#0e1116}
.err{color:var(--bad);font-size:13px;white-space:pre-wrap}
small{color:var(--dim)}
#modeTag{font-weight:700;color:var(--acc)}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;
white-space:nowrap}th{color:var(--dim);font-weight:600}
tr.bad td{color:var(--bad)}tr.warn td{color:var(--warn)}
.card{min-width:0}
/* ---- portrait 9:16 operator monitor ------------------------------------
   The wide vertical screen makes one column wasteful. Pack EVERY control
   into the top third: a full-width status ribbon (row 1) over a 3-column
   grid (row 2 — episode / session / secondary-stack). The rerun viewer then
   flex-grows to fill the bottom two-thirds. Nothing is hidden below it. */
@media (orientation:portrait){
  html,body{height:100%}
  main{display:flex;flex-direction:column;height:100vh;height:100dvh;
    max-width:100%;margin:0;padding:0;gap:0;align-items:stretch}
  .controls{flex:0 0 auto;gap:0;border-bottom:2px solid var(--line)}
  .viewer{flex:1 1 0;min-height:0;gap:0}
  /* edge-to-edge, tight cards */
  .controls .card{border:0;border-radius:0;padding:6px 10px}
  /* row 1 — status ribbon: title + pills + live numbers on one line */
  .statusband{display:flex;align-items:center;flex-wrap:wrap;gap:4px 14px;
    border-bottom:1px solid var(--line)}
  .statusband h1{font-size:14px;margin:0;white-space:nowrap}
  .statusband #pills{margin:0;display:flex;flex-wrap:wrap;gap:4px}
  .statusband .pill{margin:0}
  .statusband .live{display:flex;align-items:center;flex-wrap:wrap;gap:2px 14px;margin:0}
  .statusband .live .kv{padding:0;gap:5px;font-size:12px}
  .statusband .live .bar{width:80px;height:8px}
  .statusband .err{flex-basis:100%;margin:0}
  .statusband .err:empty{display:none}
  /* row 2 — three dense control columns with hairline dividers */
  .grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;
    align-items:start}
  .grid3>.card{border-right:1px solid var(--line)}
  .colC{display:flex;flex-direction:column;border-right:1px solid var(--line)}
  .colC>.card{border-bottom:1px solid var(--line)}
  .colC>.card:last-child{border-bottom:0}
  /* dense typography + compact touch targets (40-44px, still tappable) */
  .controls h2{font-size:11px;margin:0 0 4px;letter-spacing:.04em}
  .controls label{font-size:11px;margin:4px 0 2px}
  .controls input,.controls select{min-height:36px;padding:5px 8px;border-radius:6px}
  .controls button{min-height:40px;padding:6px 8px;border-radius:6px}
  .controls button.big{min-height:42px;font-size:13px;padding:8px}
  .controls .kv{font-size:12px;padding:1px 0}
  .controls .bar{height:8px}
  .controls small{font-size:10px}
  #eplist{max-height:76px}
  #eplist div{padding:2px 0}
  #sgEnabled{min-height:auto}
  .banner{margin:0;border-radius:0;padding:8px 10px}
  /* viewer fills the bottom two-thirds; iframe edge-to-edge, no fixed height */
  .viewer .card{flex:1;display:flex;flex-direction:column;border:0;border-radius:0;
    padding:6px 0 0}
  .viewer .card h2{margin:2px 10px 4px;font-size:11px}
  #rerun{flex:1 1 auto;width:100%;height:auto;min-height:0;border:0;border-radius:0}
  /* the verify report is normally hidden; when shown, cap it so it never
     steals the whole viewer — it scrolls inside its own box */
  #vCard{flex:0 0 auto;max-height:34vh;overflow:auto}
}
/* ---- controls-only mode (rerun runs NATIVELY on the rig display) ----------
   No embedded viewer at all: the panel is a pure control surface meant to be
   dragged to a small, freely-resized browser window. Layout is WIDTH-driven,
   not orientation-driven — one column when narrow, more columns as it widens —
   and it wins over the portrait rules above via the extra .controls-only class.
   Toggled by JS from state.viewer_mode; "web" keeps the iframe path untouched. */
body.controls-only main{display:block;height:auto;max-width:1100px;
  margin:0 auto;padding:10px}
body.controls-only .controls{gap:12px;border-bottom:0}
body.controls-only #rerunCard{display:none}   /* viewer is on the rig — no iframe */
/* keep normal card chrome regardless of orientation (undo the portrait resets) */
body.controls-only .controls .card{border:1px solid var(--line);
  border-radius:10px;padding:12px}
/* status ribbon: title + pills + live numbers wrap onto as few lines as fit */
body.controls-only .statusband{display:flex;align-items:center;flex-wrap:wrap;
  gap:4px 14px}
body.controls-only .statusband h1{font-size:15px;margin:0;white-space:nowrap}
body.controls-only .statusband #pills{margin:0;display:flex;flex-wrap:wrap;gap:4px}
body.controls-only .statusband .pill{margin:0}
body.controls-only .statusband .live{display:flex;align-items:center;
  flex-wrap:wrap;gap:2px 14px;margin:0}
body.controls-only .statusband .live .kv{padding:0;gap:5px}
body.controls-only .statusband .live .bar{width:80px;height:8px}
body.controls-only .statusband .err{flex-basis:100%;margin:0}
body.controls-only .statusband .err:empty{display:none}
/* CSS multi-column masonry: cards flow DOWN each column and the browser
   balances column heights, so columns pack densely with no ragged tails or
   empty gaps at any width (a plain auto-fit grid placed row-by-row and left
   short/long columns). Column count follows width via column-width alone — no
   breakpoints. Episode card is first in DOM, so it stays at the top of the
   first column (the record buttons never scroll out of view). */
body.controls-only .grid3{display:block;column-width:240px;column-gap:10px}
body.controls-only .colC{display:contents}
body.controls-only .grid3>.card,body.controls-only .colC>.card{
  display:block;width:100%;margin:0 0 10px;
  break-inside:avoid;-webkit-column-break-inside:avoid}
/* verify report flows below the controls; scrolls inside its own box */
body.controls-only .viewer{display:block}
body.controls-only #vCard{flex:0 0 auto;max-height:none;margin-top:12px;
  border:1px solid var(--line);border-radius:10px;padding:12px}
body.controls-only #vCard h2{margin:0 0 8px;font-size:13px}
/* very small windows: tighten spacing and let a narrower column pack 2-up */
@media (max-width:520px){
  body.controls-only main{padding:6px}
  body.controls-only .controls{gap:8px}
  body.controls-only .controls .card{padding:10px}
  body.controls-only .grid3{column-width:210px;column-gap:8px}
  body.controls-only .grid3>.card,body.controls-only .colC>.card{margin:0 0 8px}
}
</style></head><body><main>
<section class="controls">
  <!-- row 1: full-width status ribbon -->
  <div class="card statusband" id="statusCard">
    <h1>collect <small>· PHANTOM teleop + data collection</small></h1>
    <div id="pills"></div>
    <span class="pill bad" id="connPill" style="display:none">&#9888; server reconnecting…</span>
    <div class="live">
      <div class="kv"><span>loop</span><b id="tick">–</b></div>
      <div class="kv"><span>wrist |F|</span><b id="wrench">–</b></div>
      <div class="kv"><span>fingertip |F|</span><b id="tacn">–</b></div>
      <div class="kv"><span>gripper</span><b id="gripv">–</b></div>
      <div class="bar"><i id="gripbar"></i></div>
    </div>
    <div class="err" id="error"></div>
  </div>

  <div class="banner" id="sgBanner">
    <h3>&#9888; DM-TAC SAFEGUARD TRIPPED</h3>
    <div id="sgMsg" style="margin-bottom:8px"></div>
    <div>Teleop is frozen and the gripper was opened. The episode was saved as
    a failure. Remove the object / relax the scene, then:</div>
    <button class="big acc" style="margin-top:8px" onclick="cmd('resume_safeguard')">
      RESUME COLLECTION</button>
  </div>

  <!-- row 2: three dense control columns -->
  <div class="grid3">
    <div class="card" id="epCard"><h2>Episode</h2>
      <!-- idle: only Start is shown -->
      <button class="big acc" id="epStartBtn"
        onclick="epClick('start','start_stop')">&#9654; Start episode</button>
      <!-- recording: Stop + Discard -->
      <div id="epRecCtl" style="display:none">
        <button class="big" id="epStopBtn"
          onclick="epClick('stop','start_stop')">&#9209; Stop episode</button>
        <button class="big bad" style="margin-top:6px"
          onclick="epClick('stop','abort')">Discard</button>
      </div>
      <!-- finalizing / saving: disabled spinner -->
      <button class="big busy" id="epBusyBtn" disabled style="display:none">
        <span class="spinner"></span><span id="epBusyLabel">Finalizing…</span></button>
      <!-- stopped, awaiting verdict: success / fail / discard -->
      <div id="epVerdict" style="display:none">
        <div class="row">
          <button class="acc" onclick="epClick('verdict','success')">&#10003; Success</button>
          <button onclick="epClick('verdict','fail')">&#10007; Fail</button>
        </div>
        <button class="big bad" style="margin-top:6px"
          onclick="epClick('verdict','abort')">Discard</button>
      </div>
      <div id="epDetail"></div>
      <div style="margin-top:8px" class="kv"><span>episodes</span><b id="eps">0</b></div>
      <div class="bar"><i id="epbar"></i></div>
      <div class="eplist" id="eplist" style="margin-top:6px"></div>
    </div>

    <div class="card" id="wizard"><h2>Session</h2>
      <div class="row">
        <div><label>task</label><input id="task" placeholder="usbplug_insertion"></div>
        <div><label>operator</label><input id="operator"></div>
      </div>
      <label>instruction text (optional)</label><input id="text">
      <label>collection mode</label>
      <select id="mode">
        <option value="full">full — all DM-Tac modalities (PHANTOM)</option>
        <option value="lite">lite — UR3 + RGB + 6-axis force only</option>
      </select>
      <label>target episodes (0 = open-ended)</label>
      <input type="number" id="target" value="0" min="0">
      <button class="big acc" style="margin-top:10px" id="startBtn"
        onclick="startSession()">Start session</button>
      <button class="big" style="margin-top:6px;display:none" id="stopBtn"
        onclick="stopSession()">End session (auto-offload)</button>
      <button class="big busy" style="margin-top:10px;display:none" id="busyBtn"
        disabled><span class="spinner"></span><span id="busyLabel">Starting…</span></button>
      <div id="busyDetail"></div>
    </div>

    <div class="colC">
      <div class="card" id="sgCard"><h2>Safeguard</h2>
        <label><input type="checkbox" id="sgEnabled" onchange="sgApply()"
          style="width:auto;margin-right:6px">enabled — trips stop teleop + open gripper
        </label>
        <div class="row">
          <div><label>force limit, N (max 30)</label>
            <input type="number" id="sgForce" step="0.5" min="0.5"></div>
          <div><label>depth limit</label>
            <input type="number" id="sgDepth" step="0.05" min="0.05"></div>
        </div>
        <button style="margin-top:8px;width:100%" onclick="sgApply()">Apply thresholds</button>
        <small id="sgLive"></small>
      </div>

      <div class="card" id="faultCard" style="display:none;border-color:var(--bad)">
        <h2 style="color:var(--bad)">&#9888; ARM FAULT &mdash; protective stop</h2>
        <div style="margin-bottom:8px">Teleop is parked and the UR control
        script is dead. On the pendant: clear the protective stop, power on /
        release the brakes, and jog the arm somewhere safe. Then:</div>
        <button class="big acc" id="reBtn" onclick="doReengage()">RE-ENGAGE ARM</button>
        <div id="reDetail" class="sub" style="margin-top:6px"></div>
        <small>Glides from wherever the arm is now to the leader pose. The
        session stays open &mdash; just carry on with the next episode.</small>
      </div>

      <div class="card" id="calCard"><h2>Gripper calibration</h2>
        <button style="width:100%" onclick="cmd('calibrate_gripper')" id="calBtn">
          Calibrate leader ticks (3 s squeeze cycle)</button>
        <small id="calMsg"></small>
      </div>

      <div class="card" id="offCard"><h2>Offload — external drive</h2>
        <div class="kv"><span>staging</span><b id="staging" style="font-size:11px">–</b></div>
        <div class="bar" style="margin:6px 0"><i id="offbar"></i></div>
        <small id="offMsg"></small>
        <button style="width:100%;margin-top:6px" onclick="post('/api/offload',{})">
          Offload now</button>
      </div>

      <div class="card" id="pbCard"><h2>Episodes — playback &amp; verify</h2>
        <div class="row">
          <select id="epsel"></select>
          <button style="flex:0 0 auto" title="refresh episode list"
            onclick="loadEps()">&#8635;</button>
        </div>
        <div class="row" style="margin-top:6px">
          <button class="acc" onclick="pb('play')">&#9654; play</button>
          <button onclick="pb('pause')">&#9208;</button>
          <button onclick="pb('resume')">&#9655;&#9655;</button>
          <button onclick="pb('stop')">&#9209;</button>
          <select id="pbSpeed" style="flex:0 0 76px" onchange="pbSpeed()">
            <option value="0.25">0.25&times;</option>
            <option value="0.5">0.5&times;</option>
            <option value="1" selected>1&times;</option>
            <option value="2">2&times;</option>
            <option value="4">4&times;</option>
          </select>
        </div>
        <button style="width:100%;margin-top:6px" onclick="pbVerify()">
          Verify recording</button>
        <div class="kv" style="margin-top:6px"><span>playback</span>
          <b id="pbStat">idle</b></div>
        <div class="bar"><i id="pbBar"></i></div>
        <small id="pbMsg"></small>
      </div>
    </div>
  </div>
</section>

<section class="viewer">
  <!-- Embedded viewer — WEB mode only. In native mode (viewer_mode!="web") the
       whole card is removed via the .controls-only body class; the rerun window
       runs on the rig display instead. Kept intact as the web-mode fallback. -->
  <div class="card" id="rerunCard" style="padding:8px">
    <h2 style="margin:4px 6px">Live data — rerun (<span id="modeTag">–</span> mode)</h2>
    <p style="margin:0 6px 4px;font-size:11px;opacity:.6">
      Window layout: arrange as you like → rerun menu (☰) → Save blueprint →
      save it as <code>configs/rerun_blueprint.rbl</code> in the repo on the NUC.
      It is picked up on the next session start.</p>
    <iframe id="rerun" src="about:blank"></iframe>
  </div>
  <div class="card" id="vCard" style="display:none">
    <h2>Verification — <span id="vName"></span></h2>
    <div id="vVerdict" style="margin-bottom:6px"></div>
    <div id="vIssues"></div>
    <div style="overflow-x:auto"><table id="vTable"></table></div>
  </div>
</section>
</main><script>
let S={};
function $(id){return document.getElementById(id)}
function cmd(b){post('/api/cmd',{button:b})}
function post(u,body){return fetch(u,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(r=>r.json()).then(j=>{if(j.error)$('error').textContent=j.error;
  else $('error').textContent='';return j})}
// optimistic button state so the spinner shows on the click, not on the next
// SSE tick (~0.5 s later); reconciled against the real phase in render()
let pending='',pendingT=0;   // '' | 'start' | 'stop'
function startSession(){pending='start';pendingT=Date.now();render();
  post('/api/session/start',{task:$('task').value,
  operator:$('operator').value,text:$('text').value,mode:$('mode').value,
  target_episodes:+$('target').value})
  .then(j=>{if(j&&j.error){pending='';render();}})}
function stopSession(){pending='stop';pendingT=Date.now();render();post('/api/session/stop',{})}
// episode buttons: optimistic like the session button so the click flips the
// controls immediately (and a second click can't double-fire) instead of
// waiting for the next SSE tick. epPending is reconciled against episode_phase.
let epPending='',epPendingT=0;
function epClick(kind,btn){epPending=kind;epPendingT=Date.now();render();cmd(btn)}
function sgApply(){post('/api/safeguard',{enabled:$('sgEnabled').checked,
  force_limit_n:+$('sgForce').value,depth_limit:+$('sgDepth').value})
  .then(j=>{$('sgLive').textContent=j.ok?(j.live?'applied to the live session'
  :'saved — applies to the next session'):''})}
function esc(t){const d=document.createElement('div');
  d.textContent=String(t==null?'':t);return d.innerHTML}
function loadEps(){fetch('/api/episodes').then(r=>r.json()).then(j=>{
  if(j.error){$('error').textContent=j.error;return}
  const sel=$('epsel'),cur=sel.value;sel.innerHTML='';
  const groups={};
  j.episodes.forEach(e=>{const k=e.source+' · '+(e.session||'(root)');
    (groups[k]=groups[k]||[]).push(e)});
  Object.keys(groups).forEach(k=>{const og=document.createElement('optgroup');
    og.label=k;groups[k].forEach(e=>{const o=document.createElement('option');
    o.value=e.path;o.textContent=e.name+' · '+e.mode+' · '+e.size_mb+' MB'+
      (e.status!=='finalized'?' · '+e.status.toUpperCase():'');
    og.appendChild(o)});sel.appendChild(og)});
  if(cur)sel.value=cur;
  if(!sel.value&&sel.options.length)sel.selectedIndex=0;}).catch(()=>{})}
function pb(op){post('/api/playback/'+op,op==='play'?
  {path:$('epsel').value,speed:+$('pbSpeed').value}:{})}
function pbSpeed(){post('/api/playback/speed',{speed:+$('pbSpeed').value})}
function pbVerify(){post('/api/playback/verify',{path:$('epsel').value})}
let lastReport='';
function renderReport(r){
  const c=$('vCard');
  if(!r){c.style.display='none';lastReport='';return}
  const key=JSON.stringify(r);
  if(key===lastReport)return;   // keep the DOM stable so text is selectable
  lastReport=key;
  c.style.display='block';
  $('vName').textContent=r.name||'';
  $('vVerdict').innerHTML=r.ok?
    '<span class="pill on">OK — everything recorded</span>':
    '<span class="pill bad">ISSUES FOUND</span>';
  let iss=(r.issues||[]).map(i=>'<div class="err">&#10007; '+esc(i)+'</div>');
  iss=iss.concat((r.missing||[]).map(m=>r.ok?
    '<div style="color:var(--warn);font-size:13px">&#9888; missing stream: '+
      esc(m)+' (advisory — other hardware config)</div>':
    '<div class="err">&#10007; missing stream: '+esc(m)+'</div>'));
  iss=iss.concat((r.warns||[]).map(w=>
    '<div style="color:var(--warn);font-size:13px">&#9888; '+esc(w)+'</div>'));
  $('vIssues').innerHTML=iss.join('');
  let h='<tr><th>stream</th><th>rows</th><th>dur s</th><th>Hz</th>'+
    '<th>expect</th><th>shape</th><th>flags</th></tr>';
  (r.streams||[]).forEach(s=>{
    const flags=[].concat(s.issues||[],s.warns||[]).join('; ');
    const cls=(s.issues&&s.issues.length)?'bad':
      ((s.warns&&s.warns.length)?'warn':'');
    h+='<tr class="'+cls+'"><td>'+esc(s.name)+'</td><td>'+s.rows+'</td><td>'+
      s.dur_s+'</td><td>'+s.hz+'</td><td>'+(s.expect_hz||'')+'</td><td>'+
      esc(s.shape)+'</td><td>'+esc(flags)+'</td></tr>'});
  $('vTable').innerHTML=h;
}
let sgDirty=false;
['sgForce','sgDepth'].forEach(id=>$(id).addEventListener('input',()=>sgDirty=true));
let reClick=0;
function doReengage(){
  // the server needs up to ~10 s (teardown + settle + script start) and the
  // session loop is blocked meanwhile, so give feedback IMMEDIATELY rather
  // than waiting for the next 2 Hz SSE frame.
  reClick=Date.now();
  cmd('reengage');
  render();
}
function render(){
  const s=S, run=s.phase==='running';
  let pills='<span class="pill '+(run?'on':'')+'">'+s.phase+'</span>';
  if(s.recording)pills+='<span class="pill rec">&#9679; REC '+s.ep_seconds+'s</span>';
  if(s.safeguard_tripped)pills+='<span class="pill bad">SAFEGUARD</span>';
  if(s.arm_hold)pills+='<span class="pill bad">ARM: '+s.arm_hold+'</span>';
  if(s.workspace_hold)pills+='<span class="pill warn">WORKSPACE HOLD</span>';
  if(s.track_phase&&run)pills+='<span class="pill">'+s.track_phase+'</span>';
  $('pills').innerHTML=pills;
  $('tick').textContent=s.tick_hz?s.tick_hz.toFixed(1)+' Hz':'–';
  $('wrench').textContent=s.wrench_N.toFixed(1)+' N';
  $('tacn').textContent=s.tactile_force_N.toFixed(2)+' N';
  $('gripv').textContent=(100*s.gripper).toFixed(0)+' %';
  $('gripbar').style.width=(100*s.gripper)+'%';
  $('error').textContent=s.error||'';
  $('faultCard').style.display=(s.track_phase==='fault'||s.arm_hold)?'block':'none';
  // local flag bridges the <=0.5 s until the server reports reengaging=true
  const reBusy=s.reengaging||(Date.now()-reClick<1500);
  const rb=$('reBtn');
  if(rb){
    rb.disabled=reBusy;
    rb.innerHTML=reBusy
      ?'<span class="spinner"></span> RE-ENGAGING…'
      :'RE-ENGAGE ARM';
    $('reDetail').textContent=reBusy
      ?(s.busy_detail||'rebuilding the UR control script… (up to ~10 s)')
      :'';
  }
  $('sgBanner').className='banner'+(s.safeguard_tripped?' show':'');
  $('sgMsg').textContent=s.safeguard_msg;
  $('sgEnabled').checked=s.safeguard_enabled;
  if(!sgDirty){$('sgForce').value=s.safeguard_force_limit_n;
    $('sgDepth').value=s.safeguard_depth_limit}
  // session button state machine: setup -> starting -> running -> stopping.
  // the server confirms the optimistic 'pending' by moving the phase.
  if(pending==='start'&&(s.phase==='starting'||s.phase==='running'))pending='';
  // a phase still 'running' a beat later also settles a stop: the loop
  // REFUSES to end a session whose last episode has no verdict and puts the
  // phase back, and without this the spinner would hide the End-session
  // button for the full 10 s timeout. The delay keeps the optimistic
  // spinner: this same render runs immediately after the click.
  if(pending==='stop'&&(s.phase==='stopping'||s.phase==='setup'
    ||(s.phase==='running'&&Date.now()-pendingT>1200)))pending='';
  // timeout safety: if the server never confirms the phase (dead/wedged
  // server, dropped POST), drop the optimistic spinner after 10 s
  if(pending&&Date.now()-pendingT>10000)pending='';
  const busyPhase=s.phase==='starting'||s.phase==='stopping';
  const busy=busyPhase||pending!=='';
  const stopping=s.phase==='stopping'||pending==='stop';
  $('startBtn').style.display=(!busy&&!run)?'block':'none';
  $('stopBtn').style.display=(run&&!busy)?'block':'none';
  $('busyBtn').style.display=busy?'flex':'none';
  $('busyLabel').textContent=stopping?'Stopping…':'Starting…';
  $('busyDetail').textContent=s.busy_detail||'';
  // episode contour state machine: idle -> recording -> finalizing ->
  // awaiting_verdict -> saving -> idle. Only the buttons that actually apply
  // right now are shown (no dead verdict buttons on screen).
  let ep=s.episode_phase||'idle';
  if(epPending){
    const settled=(epPending==='start'&&ep!=='idle')||
      (epPending==='stop'&&ep!=='recording')||
      (epPending==='verdict'&&ep!=='awaiting_verdict');
    if(settled||Date.now()-epPendingT>1500)epPending='';
  }
  let epView=ep;
  if(epPending==='start')epView='recording';
  else if(epPending==='stop')epView='finalizing';
  else if(epPending==='verdict')epView='saving';
  // only while a session is actually live: 'idle' also covers setup and the
  // whole bring-up, and a Start there is a command with no session to run it
  $('epStartBtn').style.display=(epView==='idle'&&run)?'block':'none';
  $('epRecCtl').style.display=(epView==='recording')?'block':'none';
  const epBusy=(epView==='finalizing'||epView==='saving');
  $('epBusyBtn').style.display=epBusy?'flex':'none';
  $('epBusyLabel').textContent=(epView==='saving')?'Saving…':'Finalizing…';
  $('epVerdict').style.display=(epView==='awaiting_verdict')?'block':'none';
  $('epDetail').textContent=s.episode_detail||'';
  $('eps').textContent=s.ep_count+(s.target_episodes?' / '+s.target_episodes:'');
  $('epbar').style.width=(s.target_episodes?
    Math.min(100,100*s.ep_count/s.target_episodes):0)+'%';
  $('eplist').innerHTML=s.episodes.slice().reverse().map(e=>
    '<div class="'+e.outcome+'">'+e.name+' — '+e.outcome+
    (e.tags.length?' ['+e.tags.join(',')+']':'')+'</div>').join('');
  $('calMsg').textContent=s.calibrating?'CAPTURING — squeeze fully, then release':
    (s.calib_msg||'');
  $('calBtn').disabled=s.calibrating||!run;
  $('staging').textContent=s.staging_dir||'–';
  $('offbar').style.width=(s.offload_total?
    100*s.offload_done/s.offload_total:0)+'%';
  $('offMsg').textContent=s.offload_msg||'';
  $('offMsg').style.color=s.offload_ok?'':'var(--bad)';
  $('pbStat').textContent=(s.playback_state||'idle')+
    (s.playback_dur_s?' — '+(s.playback_episode||'')+' '+
      s.playback_pos_s.toFixed(1)+'/'+s.playback_dur_s.toFixed(1)+' s':'')+
    (s.verify_running?' · verifying…':'');
  $('pbBar').style.width=(s.playback_dur_s?
    Math.min(100,100*s.playback_pos_s/s.playback_dur_s):0)+'%';
  $('pbMsg').textContent=s.playback_msg||'';
  renderReport(s.verify_report);
  $('modeTag').textContent=s.mode;
  // native viewer mode: the panel becomes controls-only (no embedded viewer;
  // rerun runs on the rig display). "web" keeps the iframe path below.
  document.body.classList.toggle('controls-only',s.viewer_mode!=='web');
  const ifr=$('rerun');
  // web mode only: load the WASM viewer iframe once the URL is known. Skipped
  // entirely in native mode (viewer_native / controls-only) so the tab never
  // burns the ~3.5 cores the browser viewer costs on the rig.
  if(s.viewer_mode==='web'&&!s.viewer_native&&s.rerun_url&&
     ifr.dataset.url!==s.rerun_url){
    ifr.dataset.url=s.rerun_url;
    ifr.src=s.rerun_url.replaceAll('localhost',location.hostname);}
}
// SSE with auto-reconnect: the panel server can segfault-and-respawn (watchdog
// ~5 s) under us. When that happens the old EventSource dies; we retry with a
// 1s->5s backoff and, on every (re)connect, FULLY resync from /api/status and
// DROP the optimistic pending spinners — otherwise a click made just before the
// crash leaves the button stuck in "Starting…" forever (the restarted server
// reports phase=setup, which the pending reconciliation never clears).
let backoff=1000;
function setConnected(c){
  document.body.classList.toggle('disconnected',!c);
  const p=$('connPill');if(p)p.style.display=c?'none':'';
}
function resync(){
  fetch('/api/status').then(r=>r.json()).then(s=>{S=s;render()}).catch(()=>{});
}
function listen(){
  const es=new EventSource('/api/events');
  es.onopen=()=>{backoff=1000;setConnected(true);
    pending='';epPending='';resync();};
  es.onmessage=e=>{setConnected(true);S=JSON.parse(e.data);render()};
  es.onerror=()=>{es.close();setConnected(false);
    setTimeout(listen,backoff);backoff=Math.min(backoff*2,5000);};
}
fetch('/api/status').then(r=>r.json()).then(s=>{S=s;render()})
  .catch(()=>{}).finally(listen);
loadEps();
document.addEventListener('visibilitychange',()=>{if(!document.hidden)resync()});
</script></body></html>
"""
