#!/usr/bin/env python3
"""Fetch ONE episode out of the published HapticWAM datasets.

The corpora are large and three of the five repositories are packed into tar
archives (see docs/dataset_schema.md §9), so the obvious `snapshot_download`
either pulls 100 GB or spends an hour enumerating ~865k loose files. This
script resolves a single episode to the ONE file that contains it — using each
repository's `index.jsonl` / manifest, or the loose path — downloads only that
file (streaming and stopping early where the container allows it), extracts
just that episode, and prints the stream inventory of what it got.

    python tools/hub/fetch_episode.py --dataset teleop --episode first
    python tools/hub/fetch_episode.py --dataset teleop --episode ep_waffles_1785592739_002
    python tools/hub/fetch_episode.py --dataset sim --episode first --out /tmp/ep
    python tools/hub/fetch_episode.py --dataset rig --episode ep_student_Carton_1789493413_000
    python tools/hub/fetch_episode.py --dataset rollouts --episode deploy_20260813
    python tools/hub/fetch_episode.py --dataset teleop --episode first --samples

`--samples` takes the published `samples/` copy instead of touching a shard —
a few tens of MB, and the fastest way to see the format.

Bulk provisioning of a whole training corpus is tools/provision_*.sh; this
script is deliberately the single-episode path and does not duplicate them.

Auth: a read token via `HF_TOKEN` (or a cached `hf auth login`) for private
repositories; public ones need nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# repositories
# ---------------------------------------------------------------------------

TELEOP_REPO = "armteam/hapticwam-teleop-dataset"     # packed: one tar per task
TELEOP_LOOSE_REPO = "armteam/hapticwam-teleop-raw"   # the same episodes, loose
SIM_REPO = "armteam/hapticwam-sim-episodes"          # packed: one tar per episode
ROLLOUTS_REPO = "armteam/hapticwam-rollouts"         # packed: one tar per deploy day
RIG_REPO = "armteam/hapticwam-rig-episodes"          # loose

DATASETS = ("teleop", "sim", "rig", "rollouts")


@dataclass(frozen=True)
class Resolution:
    """Where one episode lives: a single archive, or a loose path prefix."""
    dataset: str
    repo: str
    kind: str            # "tar" | "loose"
    path: str            # file in the repo (tar), or the loose directory prefix
    episode: str         # episode id, "" when the unit is a whole day/take pack
    member_prefix: str = ""   # extract only members under this prefix ("" = all)
    compression: str = ""     # "zst" | "gz" | "" (plain tar)
    note: str = ""

    def describe(self) -> str:
        what = f"{self.kind} {self.repo}/{self.path}"
        if self.member_prefix:
            what += f" [{self.member_prefix}*]"
        return what


# ---------------------------------------------------------------------------
# pure resolution (unit-tested against fixture index files, no network)
# ---------------------------------------------------------------------------

_EP_RE = re.compile(r"^ep_(?P<task>.+)_(?P<epoch>\d{9,11})_(?P<idx>\d+)$")


def parse_episode_id(episode: str) -> tuple[str, str, str]:
    """`ep_<task>_<epoch>_<idx>` -> (task, epoch, idx).

    The task itself may contain underscores (`waffles_fail`), and a deploy
    episode's task segment is prefixed by the policy (`student_Carton`), so the
    split is anchored on the trailing `<epoch>_<idx>`, never on the first `_`.
    """
    m = _EP_RE.match(episode)
    if not m:
        raise ValueError(
            f"{episode!r} is not an episode id of the form ep_<task>_<epoch>_<idx>")
    return m.group("task"), m.group("epoch"), m.group("idx")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _compression_of(path: str) -> str:
    if path.endswith(".tar.zst"):
        return "zst"
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        return "gz"
    return ""


def teleop_shards(file_rows: list[dict]) -> list[str]:
    """Task shards present in the teleop repo's per-file index.jsonl."""
    return sorted(r["path"] for r in file_rows
                  if r.get("path", "").endswith(".tar.zst")
                  and "/" not in r["path"] and not r["path"].startswith("batch_"))


def resolve_teleop(episode: str, file_rows: list[dict], *,
                   archive: str | None = None) -> Resolution:
    """Episode -> its task shard.

    The teleop repo's `index.jsonl` is a per-FILE index (path/size/sha256), not
    a unit->tar map: the episode->shard rule is the task, which the episode id
    carries and `manifests_v6.tar` confirms. A shard's members are
    `<task>/<episode>/...`, so the member prefix is the same string.
    """
    shards = teleop_shards(file_rows)
    if not shards:
        raise LookupError("no <task>.tar.zst shard listed in the teleop index.jsonl")
    if episode == "first":
        shard = archive or shards[0]
        return Resolution("teleop", TELEOP_REPO, "tar", shard, "",
                          compression="zst",
                          note="first episode of the shard (streamed, stops early)")
    task, _, _ = parse_episode_id(episode)
    shard = archive or f"{task}.tar.zst"
    if archive is None and shard not in shards:
        raise LookupError(
            f"{episode}: no shard {shard!r} in {TELEOP_REPO} (have: {', '.join(shards)}). "
            "Episodes of the 2026-08-22 intake batch live in batch_20260822.tar.zst under "
            "the raw session layout — pass --archive batch_20260822.tar.zst.")
    prefix = f"{task}/{episode}/" if shard == f"{task}.tar.zst" else ""
    return Resolution("teleop", TELEOP_REPO, "tar", shard, episode,
                      member_prefix=prefix, compression="zst")


def resolve_teleop_loose(episode: str) -> Resolution:
    """The same teleop episode, loose, from the raw repo — no shard involved."""
    task, _, _ = parse_episode_id(episode)
    return Resolution("teleop", TELEOP_LOOSE_REPO, "loose",
                      f"tasks/{task}/{episode}", episode)


def resolve_sim(episode: str, unit_rows: list[dict]) -> Resolution:
    """Episode -> its per-episode tar, from a sim set's unit->tar index.jsonl.

    Rows look like {"set", "unit", "tar", "bytes", "members", "sha256",
    "task", "success", "status"}; `unit` is "<area>/<...>/<episode>".
    """
    eps = [r for r in unit_rows if r.get("task")]
    if not eps:
        raise LookupError("sim index.jsonl has no episode units (rows with a task)")
    if episode == "first":
        ok = [r for r in eps
              if r.get("success") and r.get("status") == "finalized"
              and r["unit"].startswith("tasks/")] or eps
        row = min(ok, key=lambda r: int(r.get("bytes", 0)))
    else:
        hits = [r for r in eps
                if r["unit"] == episode or r["unit"].rsplit("/", 1)[-1] == episode]
        if not hits:
            raise LookupError(f"{episode}: not in the sim index "
                              f"({len(eps)} episode units listed)")
        row = hits[0]
    unit = row["unit"].rsplit("/", 1)[-1]
    return Resolution("sim", SIM_REPO, "tar", row["tar"], unit,
                      compression=_compression_of(row["tar"]),
                      note=f"{row.get('members', '?')} members, "
                           f"{int(row.get('bytes', 0)) / 1e6:.0f} MB")


def resolve_rollouts(episode: str, day_rows: list[dict], file_rows: list[dict],
                     *, day: str | None = None) -> Resolution:
    """Episode or deploy day -> the day's tar.

    `rig_deploy/index.jsonl` is a day->tar index ({"day", "tar", "bytes",
    "takes", "members", "sha256"}); the repo-root `index.jsonl` is a per-file
    index that also lists the `zarr_rollouts/` repacks and the `.tar.zst` day
    archives. A take inside a day pack cannot be resolved from either index, so
    an `ep_*` id needs its day (`--day`).
    """
    tars = {r["tar"]: r for r in day_rows if r.get("tar")}
    for r in file_rows:
        p = r.get("path", "")
        if p.endswith((".tar", ".tar.zst")) and p not in tars and "manifest" not in p:
            tars[p] = {"tar": p, "bytes": r.get("size", 0)}

    def day_of(tar_path: str) -> str:
        m = re.search(r"(deploy_\d{8}|rollouts_[\d_]+|rig_\d{8})", tar_path)
        return m.group(1) if m else Path(tar_path).name.split(".")[0]

    if episode == "first":
        row = min(tars.values(), key=lambda r: int(r.get("bytes") or 0))
        return Resolution("rollouts", ROLLOUTS_REPO, "tar", row["tar"], "",
                          compression=_compression_of(row["tar"]),
                          note="smallest pack in the repo")
    if episode.startswith("ep_"):
        if not day:
            raise LookupError(
                f"{episode}: the rollout indexes map DAYS to tars, not takes — "
                "pass --day deploy_<YYYYMMDD> (see rig_deploy/index.jsonl), or "
                "--episode deploy_<YYYYMMDD> for the whole pack")
        target, ep = day, episode
    else:
        target, ep = episode, ""
    hits = [r for p, r in sorted(tars.items()) if day_of(p) == target]
    if not hits:
        raise LookupError(f"{target}: no pack for it in {ROLLOUTS_REPO} "
                          f"(have: {', '.join(sorted({day_of(p) for p in tars}))})")
    # prefer the plain (uncompressed, early-stoppable) repack of a day
    row = sorted(hits, key=lambda r: (_compression_of(r["tar"]) != "",
                                      int(r.get("bytes") or 0)))[0]
    return Resolution("rollouts", ROLLOUTS_REPO, "tar", row["tar"], ep,
                      member_prefix=f"{target}/{ep}/" if ep else "",
                      compression=_compression_of(row["tar"]),
                      note=(f"{row['takes']} takes in the pack" if row.get("takes")
                            else f"{int(row.get('bytes') or 0) / 1e6:.0f} MB pack"))


RIG_DIRS = ("20260915_experiment", "20260915_experiment_extra",
            "20260915_experiment_superseded")


def resolve_rig(episode: str, episode_dirs: list[str]) -> Resolution:
    """Loose repo: episode -> `<top>/<episode>`. `episode_dirs` are repo paths
    of episode directories (e.g. from list_repo_tree)."""
    if not episode_dirs:
        raise LookupError(f"no episode directories listed in {RIG_REPO}")
    if episode == "first":
        path = sorted(episode_dirs)[0]
    else:
        hits = [d for d in episode_dirs if d.rsplit("/", 1)[-1] == episode]
        if not hits:
            raise LookupError(f"{episode}: not found in {RIG_REPO}")
        path = hits[0]
    return Resolution("rig", RIG_REPO, "loose", path, path.rsplit("/", 1)[-1])


def resolve_samples(dataset: str, episode: str, sample_dirs: list[str]) -> Resolution:
    """The published `samples/` copy (loose) of a packed repository."""
    repo = {"teleop": TELEOP_REPO, "sim": SIM_REPO, "rollouts": ROLLOUTS_REPO}[dataset]
    if not sample_dirs:
        raise LookupError(f"{repo} has no samples/ folder")
    if episode == "first":
        path = sorted(sample_dirs)[0]
    else:
        hits = [d for d in sample_dirs if d.rsplit("/", 1)[-1] == episode]
        if not hits:
            raise LookupError(f"{episode}: not among the samples of {repo} "
                              f"({', '.join(sorted(d.rsplit('/', 1)[-1] for d in sample_dirs))})")
        path = hits[0]
    return Resolution(dataset, repo, "loose", path, path.rsplit("/", 1)[-1],
                      note="published sample (verbatim copy of a tar member)")


# ---------------------------------------------------------------------------
# hub access
# ---------------------------------------------------------------------------

def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _download(repo: str, path: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", local_dir=cache_dir)


def _index(repo: str, path: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    return read_jsonl(hf_hub_download(repo, path, repo_type="dataset"))


def _dirs(repo: str, prefix: str) -> list[str]:
    api = _api()
    try:
        return [e.path for e in api.list_repo_tree(repo, repo_type="dataset",
                                                   path_in_repo=prefix, recursive=False)
                if not hasattr(e, "size") or getattr(e, "size", None) is None]
    except Exception:
        return []


def _sample_dirs(repo: str) -> list[str]:
    out = []
    for d in _dirs(repo, "samples"):
        sub = _dirs(repo, d)
        out.extend(sub if sub else [d])
    return out


def _open_stream(repo: str, path: str, compression: str):
    """Streaming tar reader over a hub file: yields members without ever
    writing the archive to disk, so extraction can stop early."""
    import httpx
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, path, repo_type="dataset")
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    src, proc = None, None

    if compression:
        prog = {"zst": ["zstd", "-dc"], "gz": ["gzip", "-dc"]}[compression]
        proc = subprocess.Popen(prog, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                bufsize=1 << 20)
        sink, src = proc.stdin, proc.stdout
    else:
        import queue

        class _Pipe:
            """Minimal write-end -> read-end adapter for the plain-tar case."""

            def __init__(self):
                self.q, self.buf, self.done = queue.Queue(maxsize=8), b"", False

            def write(self, b):
                self.q.put(b)

            def close(self):
                self.q.put(None)

            def read(self, n=-1):
                while self.done is False and (n < 0 or len(self.buf) < n):
                    chunk = self.q.get()
                    if chunk is None:
                        self.done = True
                        break
                    self.buf += chunk
                out, self.buf = (self.buf, b"") if n < 0 else (self.buf[:n], self.buf[n:])
                return out

        sink = src = _Pipe()

    def pump():
        """Feed the decompressor/pipe, resuming with a Range request if the
        connection drops mid-archive (a 15 GB shard over a flaky link
        otherwise has to start over)."""
        pos = 0
        try:
            for attempt in range(1, 7):
                h = dict(headers)
                if pos:
                    h["Range"] = f"bytes={pos}-"
                try:
                    with httpx.stream("GET", url, headers=h, follow_redirects=True,
                                      timeout=120.0) as r:
                        r.raise_for_status()
                        if pos and r.status_code != 206:
                            raise RuntimeError(
                                "the server ignored the Range request — cannot resume "
                                "this transfer; rerun with --no-stream")
                        for chunk in r.iter_bytes(1 << 20):
                            sink.write(chunk)
                            pos += len(chunk)
                    return                      # transfer complete
                except (BrokenPipeError, ValueError):
                    return                      # consumer stopped early, on purpose
                except Exception as e:          # noqa: BLE001 — resume and retry
                    print(f"  (transfer interrupted at {pos / 1e6:.1f} MB: "
                          f"{type(e).__name__}: {e} — attempt {attempt}/6)",
                          file=sys.stderr)
        finally:
            try:
                sink.close()
            except Exception:
                pass

    threading.Thread(target=pump, daemon=True).start()
    tf = tarfile.open(fileobj=src, mode="r|")
    return tf, proc


def extract_from_tar(res: Resolution, out: Path, *, stream: bool = True) -> Path:
    """Extract `res.episode` (or the first episode, or the whole pack) from the
    resolved archive into `out`. Returns the episode directory."""
    out.mkdir(parents=True, exist_ok=True)
    want = res.member_prefix
    first_mode = res.episode == "" and not want
    got: Path | None = None
    cur = None

    if stream:
        tf, proc = _open_stream(res.repo, res.path, res.compression)
    else:
        local = _download(res.repo, res.path)
        tf, proc = tarfile.open(local, mode="r"), None

    try:
        for m in tf:
            name = m.name
            parts = Path(name).parts
            ep_at = next((i for i, p in enumerate(parts) if p.startswith("ep_")), None)
            if first_mode:
                if ep_at is None:
                    continue
                ep = "/".join(parts[: ep_at + 1]) + "/"
                if cur is None:
                    cur = ep
                elif ep != cur:
                    break                       # first episode complete
                want = cur
            if want and not name.startswith(want):
                if got is not None:
                    break                       # past the target: stop the transfer
                continue
            tf.extract(m, path=out, set_attrs=False)
            if got is None and ep_at is not None:
                got = out / "/".join(parts[: ep_at + 1])
    except tarfile.ReadError as e:
        raise RuntimeError(
            f"the streamed archive ended early ({e}) — the transfer could not be "
            "resumed; rerun with --no-stream to download the whole archive first") from e
    finally:
        try:
            tf.close()
        except Exception:
            pass
        if proc is not None:
            proc.kill()
    if got is None:
        raise LookupError(
            f"{res.episode or 'the first episode'} not found in {res.path} "
            f"(looked for members under {want!r})")
    return got


def download_loose(res: Resolution, out: Path) -> Path:
    from huggingface_hub import snapshot_download
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(res.repo, repo_type="dataset", local_dir=str(out),
                      allow_patterns=[f"{res.path}/**"])
    return out / res.path


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def inventory(ep: Path) -> str:
    """Print what we got: meta + one line per stream (shape, dtype, rate)."""
    import numpy as np
    lines = [f"episode {ep}"]
    meta_path = ep / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        lines.append(
            "  meta: task={task} policy={policy} success={success} status={status} "
            "tags={tags} config_hash={ch}".format(
                task=m.get("task"), policy=m.get("policy") or "-",
                success=m.get("success"), status=m.get("status"),
                tags=m.get("tags"), ch=m.get("config_hash")))
    try:
        import zarr
    except ImportError:
        lines.append("  (install zarr to see the stream inventory)")
        return "\n".join(lines)
    total = 0
    for zp in sorted(p for p in ep.iterdir() if p.is_dir() and p.suffix == ".zarr"):
        g = zarr.open_group(str(zp), mode="r")
        if "data" not in g or "ts" not in g:
            # a stream whose group was created but never written (it happens in
            # a few early deploy takes) — report it rather than crashing
            lines.append(f"  {zp.stem:<26} EMPTY (no data/ts array)")
            continue
        ts = np.asarray(g["ts"][:], dtype=np.float64)
        enc = g.attrs.get("encoding")
        shape = ((len(ts), *g.attrs["frame_shape"]) if enc == "jpeg" else g["data"].shape)
        dtype = (g.attrs.get("frame_dtype", "uint8") if enc == "jpeg" else g["data"].dtype)
        hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else float("nan")
        nbytes = sum(f.stat().st_size for f in zp.rglob("*") if f.is_file())
        total += nbytes
        lines.append(f"  {zp.stem:<26} {str(tuple(shape)):<22} {str(dtype):<8} "
                     f"{hz:7.2f} Hz  {nbytes / 1e6:7.1f} MB"
                     f"{'  [jpeg]' if enc == 'jpeg' else ''}")
    lines.append(f"  {len(lines) - 2} streams, {total / 1e6:.1f} MB")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_from_hub(args) -> Resolution:
    if args.samples:
        return resolve_samples(args.dataset, args.episode, _sample_dirs(
            {"teleop": TELEOP_REPO, "sim": SIM_REPO,
             "rollouts": ROLLOUTS_REPO}[args.dataset]))
    if args.dataset == "teleop":
        if args.loose:
            return resolve_teleop_loose(args.episode)
        return resolve_teleop(args.episode, _index(TELEOP_REPO, "index.jsonl"),
                              archive=args.archive)
    if args.dataset == "sim":
        rows: list[dict] = []
        for s in _dirs(SIM_REPO, ""):
            try:
                rows += _index(SIM_REPO, f"{s}/index.jsonl")
            except Exception:
                continue
        return resolve_sim(args.episode, rows)
    if args.dataset == "rollouts":
        return resolve_rollouts(args.episode,
                                _index(ROLLOUTS_REPO, "rig_deploy/index.jsonl"),
                                _index(ROLLOUTS_REPO, "index.jsonl"), day=args.day)
    eps = [d for top in RIG_DIRS for d in _dirs(RIG_REPO, top)]
    return resolve_rig(args.episode, eps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--episode", default="first",
                    help="episode id, a deploy day (rollouts), or 'first'")
    ap.add_argument("--out", default="hub_episode", help="output directory")
    ap.add_argument("--samples", action="store_true",
                    help="take the published samples/ copy instead of a shard")
    ap.add_argument("--loose", action="store_true",
                    help="teleop: take the loose copy from hapticwam-teleop-raw")
    ap.add_argument("--archive", help="teleop: force a specific shard")
    ap.add_argument("--day", help="rollouts: the deploy day holding the take")
    ap.add_argument("--no-stream", action="store_true",
                    help="download the whole archive before extracting")
    ap.add_argument("--dry-run", action="store_true", help="resolve only")
    args = ap.parse_args(argv)

    res = resolve_from_hub(args)
    print(f"resolved {args.episode} -> {res.describe()}"
          f"{'  (' + res.note + ')' if res.note else ''}")
    if args.dry_run:
        return 0
    out = Path(args.out).expanduser().resolve()
    ep = (download_loose(res, out) if res.kind == "loose"
          else extract_from_tar(res, out, stream=not args.no_stream))
    print(inventory(ep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
