#!/usr/bin/env python3
"""Emit ONE independent hash-pinned diagnostic overlay for frozen v1.

Never edit the base tree. Apply emitted files only to a new copy of v1. Latency
schedule and accepted-command mechanics replay are separate variants; neither
includes FINISH or newer sensors/controller code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = "tools/sim/run_waffles.py"
BASE_RUNNER_SHA = "355c6a76f00c55721ffad8744337863b6962b4777bb7df27ce5ff400dcba91a6"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one reviewed source site: {old!r}")
    return text.replace(old, new, 1)


def latency_overlay(text):
    text = replace_once(
        text,
        '    p.add_argument("--seed", type=int, default=4242)\n',
        '    p.add_argument("--anchor-latency-trace", type=Path)\n'
        '    p.add_argument("--seed", type=int, default=4242)\n',
    )
    text = replace_once(
        text,
        "    args.output = args.output.resolve()\n",
        '    if args.anchor_latency_trace and (args.mode != "policy" or args.policy_latency is not None or args.inference_delay_add_s != 0):\n'
        '        raise ValueError("Recorded latency requires policy mode and no other latency override")\n'
        "    args.output = args.output.resolve()\n",
    )
    text = replace_once(
        text,
        "    adapter = policy = audit = probe = None\n",
        "    from tools.sim.anchor_inference_timing import RecordedInferenceLatencies\n"
        "    latency_schedule = RecordedInferenceLatencies(args.anchor_latency_trace) if args.anchor_latency_trace else None\n"
        "    adapter = policy = audit = probe = None\n",
    )
    text = replace_once(
        text,
        "                                latency_s=args.policy_latency,\n",
        "                                latency_s=latency_schedule.next(replan_id, t) if latency_schedule else args.policy_latency,\n",
    )
    text = replace_once(
        text,
        "                        audit.finish_replan(\n",
        "                        if latency_schedule:\n"
        '                            proposed.diag["anchor_latency_schedule"] = latency_schedule.used[-1]\n'
        "                        audit.finish_replan(\n",
    )
    text = replace_once(
        text,
        "        writer.release()\n",
        "        writer.release()\n"
        "        if latency_schedule:\n"
        '            (args.output / "anchor_latency_schedule.json").write_text(json.dumps(latency_schedule.metadata, indent=2) + "\\n")\n',
    )
    return text


def command_overlay(text):
    text = replace_once(
        text,
        '    p.add_argument("--seed", type=int, default=4242)\n',
        '    p.add_argument("--anchor-command-trace", type=Path)\n'
        '    p.add_argument("--seed", type=int, default=4242)\n',
    )
    text = replace_once(
        text,
        "    args.output = args.output.resolve()\n",
        '    if not args.anchor_command_trace or args.mode != "policy" or not args.policy_initial_state:\n'
        '        raise ValueError("Command diagnostic requires explicit trace, measured initial state and policy initialization mode")\n'
        "    if args.policy_latency is not None or args.inference_delay_add_s != 0 or args.observation_delay_s != 0:\n"
        '        raise ValueError("No inference/delay settings apply to accepted-command mechanics replay")\n'
        "    args.output = args.output.resolve()\n",
    )
    text = replace_once(
        text,
        "    adapter = policy = audit = probe = None\n",
        "    from phantom.sim.command_replay import RecordedDriveCommands\n"
        '    recorded_commands = RecordedDriveCommands(args.anchor_command_trace, finger_limit_m=cfg["gripper"]["stroke"] / 2)\n'
        '    (args.output / "command_replay.json").write_text(json.dumps(recorded_commands.metadata, indent=2) + "\\n")\n'
        "    adapter = policy = audit = probe = None\n",
    )
    text = replace_once(
        text,
        '    try:\n        if args.mode == "policy":\n            audit = PolicyAudit(args.output)\n',
        '    try:\n        if args.mode == "policy" and recorded_commands is None:\n            audit = PolicyAudit(args.output)\n',
    )
    text = replace_once(
        text,
        "            elif not stop_after_step and rgb is not None and t + 1e-9 >= next_control:\n",
        "            elif recorded_commands is not None:\n"
        "                replay_target = recorded_commands.at(t)\n"
        "                if replay_target is not None:\n"
        "                    desired[ids], desired[fingers] = replay_target\n"
        "            elif not stop_after_step and rgb is not None and t + 1e-9 >= next_control:\n",
    )
    text = replace_once(
        text,
        '        "mode": args.mode,\n',
        '        "mode": "command_replay",\n        "command_replay": recorded_commands.metadata,\n',
    )
    text = replace_once(
        text,
        '        "tactile_model": args.tactile if args.mode == "policy" else None,\n',
        '        "tactile_model": None,\n',
    )
    text = replace_once(
        text,
        '        "wrist_model": args.wrist if args.mode == "policy" else None,\n',
        '        "wrist_model": None,\n',
    )
    return text


def prepare(base_source, output_overlay, variant, donor_source=REPO, *, dry_run=False):
    base_source, output_overlay, donor_source = map(
        Path, (base_source, output_overlay, donor_source)
    )
    if output_overlay.exists():
        raise FileExistsError("Overlay output must be new")
    if any(
        output_overlay.resolve().is_relative_to(p.resolve())
        for p in (base_source, donor_source)
    ):
        raise ValueError("Overlay must be outside protected source trees")
    text = (base_source / RUNNER).read_text()
    if hashlib.sha256(text.encode()).hexdigest() != BASE_RUNNER_SHA:
        raise ValueError("Unreviewed frozen-v1 runner source hash")
    if variant == "latency_schedule":
        updated = latency_overlay(text)
        module = "tools/sim/anchor_inference_timing.py"
        scope = "Simulated per-replan delay/action-grid diagnostic; native K-seed selection still uses measured compute latency; live inputs/physics/safety remain active"
    elif variant == "accepted_commands":
        updated = command_overlay(text)
        module = "phantom/sim/command_replay.py"
        scope = "Mechanics diagnostic only: exact causal hold of historical actual drive submissions; no inference, no new safety decisions, no object state writes after initialization; never score as a policy rollout"
    else:
        raise ValueError("Unknown independent diagnostic variant")
    outputs = {RUNNER: updated, module: (donor_source / module).read_text()}
    for name, value in outputs.items():
        compile(value, name, "exec")
    manifest = {
        "variant": "frozen_v1_" + variant,
        "base_source": str(base_source.resolve()),
        "base_runner_sha256": BASE_RUNNER_SHA,
        "output_sha256": {
            name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in outputs.items()
        },
        "scope": scope,
        "unchanged": "v1 model configuration, adapter, veto, release controller, safety implementation, geometry, initialization, physics timestep, renderer settings; no FINISH overlay",
        "required_matching": "Reuse original effective config, robot USD, measured initial state, prepared episode and explicit actual recorded horizon; compare initialization and sampled drive/physics traces before inference about causes",
        "assembly": "Copy emitted modules only onto a NEW copy of frozen base source; never merge these independent overlays or edit the original",
        "dry_run": dry_run,
    }
    if not dry_run:
        output_overlay.mkdir(parents=True)
        for name, value in outputs.items():
            path = output_overlay / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        (output_overlay / "anchor_diagnostic_overlay.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["latency_schedule", "accepted_commands"], required=True
    )
    parser.add_argument("--donor-source", type=Path, default=REPO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.base_source,
                args.output_overlay,
                args.variant,
                args.donor_source,
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
