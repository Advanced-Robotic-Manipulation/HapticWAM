#!/usr/bin/env python3
"""Two completed v6 diagnostic reviews, using the verified CPU presentation code."""

import importlib.util
import os
import sys
import time
from pathlib import Path


def main():
    os.nice(19)
    os.sched_setaffinity(0, {30, 31})
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    base = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
    previous = (
        base
        / "runs/teacher_success_anchor_v5/video_reviews/render_confirmation_reviews.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verified_presentation_batch", previous
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert (
        module.sha(previous)
        == "272a989edd01a0fcf049eddd0ea4da8ef478f812ade655c43fd897fcd807f118"
    )
    assert module.sha(module.HELPER) == module.HELPER_SHA
    module.ROOT = base / "runs/teacher_success_anchor_v6/diagnostic"
    module.DEST = module.ROOT.parent / "video_reviews/diagnostic"
    module.RUNTIME = base / "source_teacher_anchor_limiter_v6"
    module.cv2.setNumThreads(1)
    assert (
        module.read(module.ROOT / "progress.json")["status"] == "all_trials_completed"
    )
    design = module.read(module.ROOT / "campaign_snapshot.json")
    cases = [
        f"{p['id']}__{c['id']}__seed{s}"
        for p in design["policies"]
        for c in design["conditions"]
        for s in design["sampling_seeds"]
    ]
    assert len(cases) == len(set(cases)) == 2
    module.DEST.mkdir(parents=True, exist_ok=True)
    path = module.DEST / "video_manifest.json"
    if path.exists():
        raise FileExistsError("Preserve existing diagnostic presentation manifest")
    result = {
        "status": "rendering",
        "scope": "CPU presentation of both completed diagnostic cases; not a model-ranking study",
        "pid": os.getpid(),
        "started_at_unix_s": time.time(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "expected_cases": cases,
        "runtime_source": str(module.RUNTIME),
        "render_helper": str(module.HELPER),
        "render_helper_sha256": module.HELPER_SHA,
        "reused_batch_helper": str(previous),
        "reused_batch_helper_sha256": module.sha(previous),
        "batch_sha256": module.sha(Path(__file__)),
        "campaign_sha256": module.sha(module.ROOT / "campaign_snapshot.json"),
        "raw_writes": False,
        "entries": [],
        "failures": [],
    }
    module.save(path, result)
    for case in cases:
        try:
            result["entries"].append(module.render(case))
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            AssertionError,
            module.subprocess.SubprocessError,
            module.cv2.error,
        ) as error:
            result["failures"].append({"case_id": case, "error": repr(error)})
        module.save(path, result)
        print(
            case,
            "rendered" if not result["failures"] else "inspect failures",
            flush=True,
        )
    result["status"] = "complete" if not result["failures"] else "render_failure"
    result["finished_at_unix_s"] = time.time()
    result["total_video_bytes"] = sum(e["video_bytes"] for e in result["entries"])
    result["total_decoded_frames"] = sum(e["decoded_frames"] for e in result["entries"])
    module.save(path, result)
    print(path, result["status"], flush=True)
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
