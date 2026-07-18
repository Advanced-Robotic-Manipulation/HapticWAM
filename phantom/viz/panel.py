"""Web control panel — the operator's single entry point for data collection.

Stdlib-only. Serves a self-contained dark-UI SPA (Russian operator-facing
labels) with three phases driven by SSE:

  setup    session wizard: rig summary + pre-flight checklist, task picker,
           operator, teleop device, episode target -> "Start session"
  running  live view: status, progress toward the target, episode log,
           embedded rerun viewer, the full episode-control button set
  (back to setup with a last-session summary when the session ends)

Wiring: buttons POST into a thread-safe queue the record loop merges exactly
like keyboard hotkeys; session start/stop delegate to a SessionRunner
(viz/session.py) when one is attached — without it (record_episodes --panel)
the panel starts in phase "running" and only the live view is used.

Binds 127.0.0.1 by default; --host 0.0.0.0 for a rig tablet (the buttons
move a robot — trusted LAN only).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

_BUTTONS = ("start_stop", "success", "fail", "failure_tag", "abort", "zero_ft", "quit")
PHASES = ("setup", "starting", "running", "stopping")


@dataclass
class PanelState:
    """Thread-safe status snapshot, updated by the record loop each tick."""
    phase: str = "setup"
    error: str = ""
    task: str = ""
    text: str = ""
    operator: str = ""
    teleop: str = ""
    recording: bool = False
    episode: str = ""
    ep_count: int = 0
    target_episodes: int = 0
    episodes: list = field(default_factory=list)   # [{name, outcome, tags, t}]
    out_dir: str = ""
    safety_hold: bool = False
    workspace_hold: bool = False   # streamer frozen at the workspace boundary
    last_safety: str = ""
    wrench_N: float = 0.0
    gripper: float = 0.0
    tick_hz: float = 0.0
    rerun_url: str = ""
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
            self.safety_hold = False
            self.last_safety = ""
            self.episode = ""
            self.error = ""
            self.rerun_url = ""
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            d = {k: getattr(self, k) for k in self.__dataclass_fields__
                 if not k.startswith("_")}
            d["episodes"] = list(d["episodes"])
        d["ep_seconds"] = round(time.time() - d.pop("t_ep_start"), 1) if d["recording"] else 0.0
        return d


class PanelServer:
    def __init__(self, state: PanelState, *, host: str = "127.0.0.1", port: int = 8788,
                 runner=None, setup_provider=None):
        """runner: object with start_session(cfg: dict) / stop_session();
        setup_provider: () -> dict for the /api/setup payload (rig summary)."""
        self.state = state
        self.host = host
        self.port = port
        self.runner = runner
        self.setup_provider = setup_provider
        self._cmds: queue.Queue[str] = queue.Queue()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def pop_buttons(self) -> dict[str, bool]:
        """Drain queued panel commands — merged like keyboard buttons."""
        out: dict[str, bool] = {}
        while True:
            try:
                out[self._cmds.get_nowait()] = True
            except queue.Empty:
                return out

    def push_button(self, name: str) -> None:
        self._cmds.put(name)

    # ------------------------------------------------------------------
    def start(self) -> None:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):        # quiet HTTP log
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
                    return json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return {}

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
                    payload = panel.setup_provider() if panel.setup_provider else {}
                    self._json(payload)
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
                    if btn in _BUTTONS:
                        panel._cmds.put(btn)
                        self._json({"ok": True, "button": btn})
                    else:
                        self._json({"error": f"unknown button {btn!r}"}, 400)
                elif self.path == "/api/session/start":
                    if panel.runner is None:
                        self._json({"error": "no session runner (started via "
                                             "record_episodes CLI?)"}, 409)
                        return
                    cfg = self._body()
                    err = panel.runner.start_session(cfg)
                    self._json({"ok": True} if err is None else {"error": err},
                               200 if err is None else 409)
                elif self.path == "/api/session/stop":
                    if panel.runner is None:
                        self._json({"error": "no session runner"}, 409)
                    else:
                        panel.runner.stop_session()
                        self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="panel-http")
        self._thread.start()
        log.info("control panel at %s", self.url)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


# ---------------------------------------------------------------------------
# the page (self-contained: no CDNs, works offline on the rig)
# ---------------------------------------------------------------------------

_PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHANTOM · сбор данных</title>
<style>
:root{--bg:#0b0e14;--card:#131826;--edge:#1e2638;--tx:#dbe2f0;--dim:#7d8aa5;
 --acc:#5aa9ff;--rec:#ff4d6a;--ok:#3ddc97;--warn:#ffb454;font-size:15px}
*{box-sizing:border-box;margin:0}
[hidden]{display:none!important}
body{background:var(--bg);color:var(--tx);height:100vh;display:flex;flex-direction:column;
 font:400 1rem/1.5 -apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif}
header{display:flex;align-items:center;gap:14px;padding:10px 18px;
 border-bottom:1px solid var(--edge);background:linear-gradient(180deg,#101625,#0b0e14)}
header h1{font-size:1.05rem;font-weight:650;letter-spacing:.14em}
header h1 b{color:var(--acc)}
.meta{color:var(--dim);font-size:.85rem}
.pill{margin-left:auto;padding:4px 14px;border-radius:99px;font-weight:650;font-size:.8rem;
 letter-spacing:.08em;background:#1c2436;color:var(--dim)}
.pill.rec{background:rgba(255,77,106,.14);color:var(--rec);animation:pulse 1.1s infinite}
.pill.hold{background:rgba(255,180,84,.15);color:var(--warn)}
.pill.ok{background:rgba(61,220,151,.12);color:var(--ok)}
@keyframes pulse{50%{opacity:.45}}
.card{background:var(--card);border:1px solid var(--edge);border-radius:12px;padding:14px 16px}
.card h2{font-size:.68rem;font-weight:650;letter-spacing:.16em;color:var(--dim);
 text-transform:uppercase;margin-bottom:10px}
.kv{display:flex;justify-content:space-between;font-size:.88rem;padding:2px 0;gap:10px}
.kv span:first-child{color:var(--dim);white-space:nowrap}
.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
button{padding:10px 12px;border:1px solid var(--edge);border-radius:10px;background:#1a2133;
 color:var(--tx);font:600 .84rem inherit;cursor:pointer;transition:.15s}
button:hover{border-color:var(--acc);transform:translateY(-1px)}
button:disabled{opacity:.45;cursor:default;transform:none}
button.primary{background:linear-gradient(135deg,#2563eb,#38bdf8);border:none;color:#fff;
 font-size:1rem;padding:14px}
button.primary.stop{background:linear-gradient(135deg,#e11d48,#ff4d6a)}
button.danger{color:var(--rec)} button.ghost{color:var(--dim)}
input,select{width:100%;padding:10px 12px;border:1px solid var(--edge);border-radius:10px;
 background:#0e1420;color:var(--tx);font:inherit}
input:focus,select:focus{outline:none;border-color:var(--acc)}
label{display:block;font-size:.78rem;color:var(--dim);margin:10px 0 4px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.74rem;font-weight:650}
.badge.mock{background:rgba(90,169,255,.14);color:var(--acc)}
.badge.real{background:rgba(61,220,151,.12);color:var(--ok)}
.badge.warn{background:rgba(255,180,84,.15);color:var(--warn)}
.banner{border-radius:12px;padding:12px 16px;font-size:.9rem;margin-bottom:12px}
.banner.err{background:rgba(255,77,106,.1);border:1px solid rgba(255,77,106,.35);color:#ffb3c0}
.banner.warn{background:rgba(255,180,84,.1);border:1px solid rgba(255,180,84,.3);color:var(--warn)}
.banner.ok{background:rgba(61,220,151,.08);border:1px solid rgba(61,220,151,.3);color:var(--ok)}
main{flex:1;min-height:0;overflow:auto}
/* setup */
#setup{max-width:1020px;margin:0 auto;padding:22px;display:grid;
 grid-template-columns:1fr 1.25fr;gap:14px;align-content:start}
#setup .full{grid-column:1/-1}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px}
.chip{padding:8px 14px;border-radius:99px;border:1px solid var(--edge);background:#0e1420;
 cursor:pointer;font-size:.84rem;color:var(--dim)}
.chip.on{border-color:var(--acc);color:var(--tx);background:rgba(90,169,255,.12)}
.check{display:flex;gap:10px;align-items:flex-start;font-size:.86rem;padding:5px 0;color:var(--dim)}
.check::before{content:"☐";color:var(--acc)}
.spin{width:44px;height:44px;border:4px solid var(--edge);border-top-color:var(--acc);
 border-radius:50%;margin:60px auto 16px;animation:rot 1s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
/* run */
#run{height:100%;display:grid;grid-template-columns:300px 1fr;min-height:0}
#run aside{padding:14px;display:flex;flex-direction:column;gap:12px;overflow-y:auto;
 border-right:1px solid var(--edge)}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.btns .wide{grid-column:1/-1}
.bar{height:8px;border-radius:99px;background:#0e1420;overflow:hidden;margin:8px 0 4px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#2563eb,#38bdf8);width:0}
canvas{width:100%;height:46px;display:block}
#eplist{max-height:180px;overflow-y:auto;font-size:.82rem}
#eplist div{display:flex;gap:8px;padding:3px 0;border-bottom:1px solid #10151f}
#viewer{position:relative;background:#05070c}
#viewer iframe{width:100%;height:100%;border:0}
#noviz{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
 color:var(--dim);font-size:.9rem;text-align:center;padding:30px}
.safety{font-size:.8rem;color:var(--warn);min-height:1.2em;word-break:break-all}
.hint{font-size:.78rem;color:var(--dim)}
.center{max-width:520px;margin:0 auto;padding:40px;text-align:center;color:var(--dim)}
/* ---- portrait monitor / rotated rig display ---- */
@media (orientation:portrait){
 #run{grid-template-columns:1fr;grid-template-rows:1fr auto}
 #viewer{order:-1;min-height:320px}
 #run aside{border-right:0;border-top:1px solid var(--edge);
  display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  align-content:start;gap:12px;max-height:46vh;overflow-y:auto}
 #run aside .card{margin:0}
 #eplist{max-height:130px}
}
@media (max-width:860px){
 #setup{grid-template-columns:1fr}
 header{flex-wrap:wrap;gap:8px}
}
</style></head><body>
<header><h1><b>PHANTOM</b> СБОР ДАННЫХ</h1>
 <span class="meta" id="who">–</span><span class="pill" id="pill">ПОДГОТОВКА</span>
 <button class="ghost danger" id="hend" hidden style="margin-left:10px"
   onclick="if(confirm('Завершить сессию? Робот остановится.'))endSession()">✕ Завершить сессию</button></header>
<main>

<!-- ============ SETUP ============ -->
<div id="setup" hidden>
 <div class="full" id="banners"></div>
 <div class="card">
  <h2>Стенд</h2><div id="rig">загрузка…</div>
  <h2 style="margin-top:14px">Перед стартом</h2>
  <div class="check">Робот в Remote Control, зона свободна</div>
  <div class="check">Сенсоры на РАЗНЫХ USB-хабах, камера наведена, свет ок</div>
  <div class="check">Экзоскелет надет и подключён (если Echo)</div>
  <div class="check">Пады сенсоров чистые, ничего их не касается</div>
 </div>
 <div class="card">
  <h2>Новая сессия</h2>
  <label>Задача</label><div class="chips" id="chips"></div>
  <input id="f_task" placeholder="или своё имя задачи (латиницей)">
  <label>Инструкция (language instruction, англ.)</label>
  <input id="f_text" placeholder="по умолчанию = имя задачи">
  <label>Оператор</label><input id="f_op" placeholder="имя / инициалы">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
   <div><label>Устройство</label><select id="f_dev">
    <option value="echo">Echo экзоскелет</option>
    <option value="spacemouse">SpaceMouse</option>
    <option value="keyboard">Клавиатура (терминал)</option>
    <option value="none">Без движения (contact play / мониторинг)</option>
   </select></div>
   <div><label>Цель, эпизодов</label><input id="f_target" type="number" value="150" min="0"></div>
  </div>
  <button class="primary" id="bgo" style="width:100%;margin-top:16px"
    onclick="startSession()">▶ Запустить сессию</button>
  <div class="hint" style="margin-top:8px">Поднимет робота, сенсоры, запись и
   визуализацию. Дальше — только кнопки записи. Запись эпизода включается кнопкой
   здесь, пробелом в терминале или тумблером на Echo.</div>
 </div>
</div>

<!-- ============ STARTING ============ -->
<div id="starting" hidden><div class="center">
 <div class="spin"></div><h3 style="color:var(--tx)">Поднимаю стенд…</h3>
 <p>робот · сенсоры · камера · safety · rerun — обычно 5–20 секунд</p></div></div>

<!-- ============ RUN ============ -->
<div id="run" hidden>
<aside>
 <div class="card"><h2>Эпизод</h2>
  <div class="btns">
   <button class="primary wide" id="bstart" onclick="cmd('start_stop')">● Начать запись</button>
   <button onclick="cmd('success')">✓ Успех</button>
   <button onclick="cmd('fail')">✗ Неудача</button>
   <button onclick="cmd('failure_tag')">⚑ Метка «намеренный провал»</button>
   <button class="danger" onclick="if(confirm('Выбросить текущий эпизод?'))cmd('abort')">Выбросить</button>
  </div></div>
 <div class="card"><h2>Прогресс <span class="mono" id="prog" style="float:right;color:var(--tx)"></span></h2>
  <div class="bar"><i id="pbar"></i></div>
  <div class="kv"><span>эпизод</span><span class="mono" id="ep">–</span></div>
  <div class="kv"><span>время</span><span class="mono" id="el">–</span></div>
  <div class="kv"><span>цикл</span><span class="mono" id="hz">–</span></div>
  <div class="safety" id="saf"></div></div>
 <div class="card"><h2>Журнал эпизодов</h2><div id="eplist"><i class="hint">пока пусто</i></div></div>
 <div class="card"><h2>Запястье |F| (оценка) <span class="mono" id="fN" style="float:right;color:var(--tx)"></span></h2>
  <canvas id="spark" width="520" height="92"></canvas>
  <div class="kv"><span>гриппер</span><span class="mono" id="gp">–</span></div></div>
 <div class="card"><div class="btns">
   <button class="ghost" onclick="cmd('zero_ft')">Обнулить F/T</button>
   <button class="ghost danger" onclick="if(confirm('Завершить сессию? Робот остановится.'))endSession()">
    Завершить сессию</button>
 </div></div>
</aside>
<div id="viewer"><div id="noviz">rerun-вьюер недоступен<br>(pip install 'phantom[viz]')</div></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const TASKS=["fragile_grasp","slippery_grasp_place","insertion","surface_wipe","inhand_regrasp","contact_play"];
const OUT={success:["✓","var(--ok)"],fail:["✗","var(--rec)"],saved:["●","var(--acc)"],
 discarded:["∅","var(--dim)"],safety_stop:["⚠","var(--warn)"]};
const hist=new Array(140).fill(0);let framed=false,phase="",epCount=-1;
function cmd(b){fetch('/api/cmd',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({button:b})})}
function endSession(){fetch('/api/session/stop',{method:'POST'})}
function startSession(){
 const cfg={task:$('f_task').value.trim(),text:$('f_text').value.trim(),
  operator:$('f_op').value.trim(),teleop:$('f_dev').value,
  target_episodes:+$('f_target').value||0};
 if(!cfg.task){$('f_task').focus();$('f_task').style.borderColor='var(--rec)';return}
 $('bgo').disabled=true;
 fetch('/api/session/start',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(cfg)}).then(r=>r.json()).then(j=>{
   if(j.error){alert(j.error);$('bgo').disabled=false}})
  .catch(()=>{alert('нет связи с панелью');$('bgo').disabled=false})}
function chips(){$('chips').innerHTML='';TASKS.forEach(t=>{
 const c=document.createElement('span');c.className='chip';c.textContent=t;
 c.onclick=()=>{$('f_task').value=t;[...$('chips').children].forEach(x=>
  x.classList.toggle('on',x===c))};$('chips').appendChild(c)})}
function loadSetup(){fetch('/api/setup').then(r=>r.json()).then(s=>{
 if(!s||!s.rig_name)return;
 const mock=s.mode==="mock";
 let h=`<div class="kv"><span>риг</span><span class="mono">${s.rig_name}</span></div>
 <div class="kv"><span>режим</span><span><span class="badge ${mock?'mock':'real'}">
  ${mock?'ТРЕНИРОВКА (mock)':'БОЕВОЙ (real)'}</span></span></div>
 <div class="kv"><span>рука</span><span class="mono">${s.arm}</span></div>
 <div class="kv"><span>сенсоры</span><span class="mono">${s.sensors.join(' · ')}</span></div>
 <div class="kv"><span>папка</span><span class="mono" style="font-size:.72rem">${s.out_root}</span></div>
 <div class="kv"><span>диск</span><span class="mono">${s.disk_free_gb} GB свободно</span></div>
 <div class="kv"><span>rerun</span><span>${s.viz_available?'<span class="badge real">готов</span>'
  :'<span class="badge warn">не установлен</span>'}</span></div>`;
 $('rig').innerHTML=h;
 let b='';
 if(!mock&&!s.bench_verified)b+=`<div class="banner warn">⚠ Железо НЕ прошло day-1 bench
  (meta.bench_verified=false) — сначала docs/hardware_bench_day1.md</div>`;
 if(!s.echo_configured)b+=`<div class="banner warn">Echo не настроен в конфиге —
  доступны SpaceMouse/клавиатура</div>`;
 $('banners').innerHTML=b+$('banners').innerHTML})}
function draw(){const c=$('spark'),x=c.getContext('2d'),W=c.width,H=c.height;
 x.clearRect(0,0,W,H);const m=Math.max(1,...hist);
 x.beginPath();hist.forEach((v,i)=>{const px=i/(hist.length-1)*W,py=H-3-(v/m)*(H-8);
 i?x.lineTo(px,py):x.moveTo(px,py)});
 x.strokeStyle='#5aa9ff';x.lineWidth=2;x.stroke();
 x.lineTo(W,H);x.lineTo(0,H);x.closePath();x.fillStyle='rgba(90,169,255,.12)';x.fill()}
function setPhase(p,s){
 if(p===phase)return;phase=p;
 $('setup').hidden=p!=='setup';$('starting').hidden=p!=='starting';
 $('run').hidden=!(p==='running'||p==='stopping');
 $('hend').hidden=!(p==='running'||p==='stopping');
 // browser Back must not leave the app mid-session: park one history entry
 // and swallow popstate while a session is live (the real "back" is
 // «Завершить сессию» — the page state is server-driven, not URL-driven)
 if(p==='running'&&!history.state)history.pushState({run:1},'');
 if(p==='setup'){$('bgo').disabled=false;
  let b='';
  if(s.error)b+=`<div class="banner err">⛔ Сессия не стартовала: ${s.error}</div>`;
  if(s.episodes.length)b+=`<div class="banner ok">Сессия завершена:
   ${s.episodes.length} эпизодов (${s.episodes.filter(e=>e.outcome==='success').length} успешных)
   → ${s.out_dir}</div>`;
  $('banners').innerHTML=b;loadSetup()}
 if(p!=='running')framed=false}
function apply(s){
 $('who').textContent=`${s.task||'–'} · ${s.teleop||'–'} · ${s.operator||'–'}`;
 const p=$('pill');
 const ph={setup:['ПОДГОТОВКА',''],starting:['ЗАПУСК','ok'],stopping:['ОСТАНОВКА','hold'],
  running:s.safety_hold?['SAFETY HOLD','hold']:s.recording?['ЗАПИСЬ','rec']:['ГОТОВ','ok']}[s.phase]||['–',''];
 p.textContent=ph[0];p.className='pill '+ph[1];
 setPhase(s.phase,s);
 if(s.phase!=='running'&&s.phase!=='stopping')return;
 const b=$('bstart');b.classList.toggle('stop',s.recording);
 b.textContent=s.recording?'■ Стоп и сохранить':'● Начать запись';
 $('ep').textContent=s.episode||'–';
 $('el').textContent=s.recording?s.ep_seconds.toFixed(1)+' с':'–';
 $('hz').textContent=s.tick_hz.toFixed(1)+' Гц';
 $('gp').textContent=(s.gripper*100).toFixed(0)+' %';
 $('fN').textContent=s.wrench_N.toFixed(1)+' Н';
 const n=s.episodes.filter(e=>e.outcome!=='discarded').length;
 $('prog').textContent=s.target_episodes?`${n} / ${s.target_episodes}`:`${n}`;
 $('pbar').style.width=(s.target_episodes?Math.min(100,100*n/s.target_episodes):0)+'%';
 $('saf').textContent=s.safety_hold?('⚠ '+(s.last_safety==='protective_stop'
  ?'Защитный стоп: снимите его на пульте робота — продолжится само'
  :'Перегрузка ('+s.last_safety+'): уберите нагрузку — продолжится само'))
  :s.workspace_hold?'🧱 Рука у границы рабочей зоны — ведите экзоскелет обратно':'';
 if(s.episodes.length!==epCount){epCount=s.episodes.length;
  $('eplist').innerHTML=s.episodes.slice().reverse().map(e=>{
   const[o,c]=OUT[e.outcome]||['?','var(--dim)'];
   return`<div><span style="color:${c}">${o}</span><span class="mono">${e.name||'—'}</span>
    <span class="hint">${e.tags.join(',')}</span></div>`}).join('')||'<i class="hint">пока пусто</i>'}
 hist.push(s.wrench_N);hist.shift();draw();
 if(s.rerun_url&&!framed){framed=true;$('viewer').querySelectorAll('iframe').forEach(f=>f.remove());
  // the backend advertises localhost URLs; from a remote browser (tablet /
  // laptop over tailscale) both the viewer page AND the gRPC stream URI in
  // its query param must point at the panel's host instead
  const u=s.rerun_url.replaceAll('localhost',location.hostname)
                     .replaceAll('127.0.0.1',location.hostname);
  const f=document.createElement('iframe');f.src=u;f.allow='fullscreen';
  $('viewer').appendChild(f);const nv=$('noviz');if(nv)nv.style.display='none'}}
window.addEventListener('popstate',()=>{
 if(phase==='running'||phase==='stopping'){history.pushState({run:1},'');
  $('saf').textContent='Сессия идёт — выход через «Завершить сессию»'}});
// survive panel restarts: EventSource auto-reconnects, but also re-pull
// status on visibility return so a stale tab picks the truth back up
document.addEventListener('visibilitychange',()=>{if(!document.hidden)
 fetch('/api/status').then(r=>r.json()).then(apply).catch(()=>{})});
chips();loadSetup();
new EventSource('/api/events').onmessage=e=>apply(JSON.parse(e.data));
fetch('/api/status').then(r=>r.json()).then(apply);
</script></body></html>
"""
