"""The operator-facing artifacts must agree with the code they drive.

Three classes of divergence have each already cost a real session or a paid
rental, and none of them was caught by a test:

1. `tools/provision_v5.sh` prints the command that spends the FT-A rental.
   `e06c33b` retracted `--contact-self-forcing` from the bundle in
   `docs/training_playbook.md` and touched nothing else, so the script kept
   launching it — and two tests PINNED the divergence in place. A flag that no
   longer exists in `train_teacher`'s parser is the same failure with a louder
   crash.
2. `docs/rig_session_v5.md` is the page the operator reads mid-session. It
   quoted the offline numbers E13 was written to retire (`20.7 -> 17.4`) and
   the pre-F17 z-floor table, against a binary that prints `zfloor:32mm`.
3. The documented `EXTRA` lines are copy-pasted verbatim into a rig shell. Arm
   B's line omitted `--max-replans 200`, so every arm-B episode ended at
   `replan_cap` after 7.2 s against a 35 s arm A — the A/B compared two
   different experiments.

So: parse the artifacts, parse the parsers, and compare.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROVISION = REPO / "tools" / "provision_v5.sh"
PLAYBOOK = REPO / "docs" / "training_playbook.md"
RIG_DOC = REPO / "docs" / "rig_session_v5.md"


def _flags(parser) -> set[str]:
    out: set[str] = set()
    for a in parser._actions:
        out.update(a.option_strings)
    return out


def _launch_block() -> str:
    """The echoed launch COMMAND — comment lines excluded."""
    src = PROVISION.read_text(encoding="utf-8")
    return src.split("READY. Launch")[1].split('echo "  #')[0]


def _smoke_block() -> str:
    src = PROVISION.read_text(encoding="utf-8")
    return (src.split("2-step REAL training smoke")[1]
               .split("READY. Launch")[0].split("\n# ablation only")[0])


def _cli_flags(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", text))


# ---------------------------------------------------------------------------
# 1. provision_v5.sh vs train_teacher's parser and the playbook
# ---------------------------------------------------------------------------

def _train_teacher_flags(monkeypatch) -> set[str]:
    """The REAL parser `train_teacher.main` builds — half its flags are added
    in `main` itself, not in `add_common_args`, so it has to be captured."""
    import argparse

    from phantom.train import train_teacher as TT
    seen: list[argparse.ArgumentParser] = []

    def _capture(self, *a, **k):
        seen.append(self)
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    with pytest.raises(SystemExit):
        TT.main([])
    assert seen, "train_teacher.main did not build a parser"
    return _flags(seen[0])


def test_every_flag_the_rental_launches_exists_in_the_parser(monkeypatch):
    """A retired or renamed flag in the printed launch line is a crash after
    the ~100 GB pull, on a rented H100, with the operator reading a script."""
    known = _train_teacher_flags(monkeypatch)
    assert "--contact-nll-beta" in known, "the parser under test is the wrong one"
    for block, what in ((_launch_block(), "launch line"),
                        (_smoke_block(), "provisioning smoke")):
        unknown = sorted(_cli_flags(block) - known)
        assert not unknown, f"{what} passes flags train_teacher does not accept: {unknown}"


def test_the_launch_line_is_the_playbook_bundle_and_nothing_it_retracted():
    """`docs/training_playbook.md`'s FT-A code block is the bundle of record.
    The script must carry all of it and none of the flags the playbook marks
    as NOT in the bundle (E9_premise_test.md retracted --contact-self-forcing;
    --contact-nll-detach-weight and --no-wrist-region-mse were never in it)."""
    book = PLAYBOOK.read_text(encoding="utf-8")
    bundle = book.split("### FT-A — the recommended objective bundle")[1]
    code = bundle.split("```bash")[1].split("```")[0]
    want = _cli_flags(code)
    assert "--contact-nll-beta" in want and "--acc-two-pass" in want, code

    for block, what in ((_launch_block(), "launch line"),
                        (_smoke_block(), "provisioning smoke")):
        got = _cli_flags(block)
        assert not (want - got), (
            f"{what} is missing playbook FT-A flags: {sorted(want - got)}")
        for retracted in ("--contact-self-forcing", "--contact-nll-detach-weight",
                          "--no-wrist-region-mse"):
            assert retracted not in got, (
                f"{retracted} is NOT in the recommended bundle "
                f"(docs/training_playbook.md) but the {what} spends the run on it")

    # ... and the retraction must be traceable from the script itself
    src = PROVISION.read_text(encoding="utf-8")
    assert "--contact-self-forcing" in src, "keep it as a commented ablation line"
    assert "E9_premise_test.md" in src, "cite the evidence for the retraction"


def test_the_launch_line_carries_the_playbook_values_not_just_the_flags():
    launch = _launch_block()
    for flag in ("--contact-nll-beta 0.5", "--ema-decay 0.995",
                 "--cond-dropout 0", "--event-band-weight 0"):
        assert flag in launch, f"{flag} missing from the printed FT-A launch line"


# ---------------------------------------------------------------------------
# 2. the rig doc quotes no retired number
# ---------------------------------------------------------------------------

RETIRED = {
    # E13_rescore.md replaced these; the doc must link there instead
    "17.4": "endpoint err (retired: val124 4-seed says 18.23)",
    "20.7": "endpoint err (retired: val124 4-seed says 20.82)",
    "14.9": "new-batch holdout (retired: re-derives as 13.78)",
    "1.72": "commit ratio (retired: 1.28)",
    "1.43": "commit ratio (retired: 1.10)",
    # the pre-F17 z-floor table, against a binary that prints zfloor:32mm
    "waffles 42": "z floor (F17: 31.5 mm)",
    "Carton 66,": "z floor (F17: 66.1 mm)",
    "whiteboard 65": "z floor (F17: 57.5 mm)",
    "egg 50": "z floor (F17: 49.5 mm)",
}


def test_the_rig_doc_quotes_no_retired_number():
    text = RIG_DOC.read_text(encoding="utf-8")
    bad = {k: why for k, why in RETIRED.items() if k in text}
    assert not bad, f"docs/rig_session_v5.md still quotes retired numbers: {bad}"


def test_the_rig_doc_points_at_the_rescore():
    text = RIG_DOC.read_text(encoding="utf-8")
    assert "E13_rescore.md" in text, (
        "the operator page must link the correction it was re-scored from")
    for n in ("20.82", "18.23", "13.78"):
        assert n in text, f"the re-scored figure {n} is not on the operator page"


def test_the_rig_doc_z_floor_table_matches_start_poses_yaml():
    """`resolve_z_floor` = `tcp_z_min` - 10 mm, straight from the config the
    binary loads. F17 moved waffles by 10.5 mm and whiteboard by 7.1 mm."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((REPO / "configs" / "start_poses.yaml").read_text())
    tasks = cfg.get("tasks", cfg)
    text = RIG_DOC.read_text(encoding="utf-8")
    seen = 0
    for task, v in tasks.items():
        if not isinstance(v, dict) or "tcp_z_min" not in v:
            continue
        floor = round(float(v["tcp_z_min"]) * 1000 - 10, 1)
        assert re.search(rf"\|\s*{re.escape(task)}\s*\|.*\|\s*\*\*{floor:.1f}\*\*\s*\|",
                         text), f"{task}: the doc's z-floor row is not {floor:.1f} mm"
        seen += 1
    assert seen == 4, f"expected 4 tasks in start_poses.yaml, found {seen}"


# ---------------------------------------------------------------------------
# 3. the documented EXTRA lines parse into what the prose claims
# ---------------------------------------------------------------------------

def _extra_lines() -> list[str]:
    """Every `EXTRA="..."` in the rig doc, in document order."""
    return re.findall(r'EXTRA="([^"]*)"', RIG_DOC.read_text(encoding="utf-8"))


def _parse_extra(extra: str):
    from phantom.scripts.run_deploy import build_parser
    # EXTRA is appended after the GO script's own flags, so it parses as a
    # tail of the real command line
    return build_parser().parse_args(
        ["--system", "teacher", "--task", "waffles", *shlex.split(extra)])


def test_every_documented_extra_line_parses_through_run_deploy():
    lines = _extra_lines()
    assert len(lines) >= 3, lines
    for extra in lines:
        _parse_extra(extra)          # argparse SystemExit == an unusable recipe


def test_arm_b_reaches_the_wall_clock_budget_it_documents():
    """C1 / VALIDATION_0830 P0 #4. `PlannerLoop.run` checks the replan COUNT
    before the wall clock, so `--max-episode-s 35` is dead at the default 40
    replans: at --nfe 1 (172 ms) the episode ends at `replan_cap` after 7.2 s
    against a 35 s arm A. Only a raised --max-replans reaches the budget."""
    arm_b = [e for e in _extra_lines() if "--terminal-veto" in e]
    assert len(arm_b) == 1, f"expected exactly one arm-B EXTRA line, got {arm_b}"
    a = _parse_extra(arm_b[0])
    assert a.nfe == 1 and a.terminal_veto and a.parity_fixes and a.k_seeds == 4
    assert a.max_episode_s == 35.0
    assert a.max_replans >= 200, (
        "arm B's EXTRA must raise --max-replans (>=200): at nfe=1 the default "
        f"{a.max_replans} replans is a ~7 s episode, not the documented 35 s")
    # the prose must not contradict the line
    text = RIG_DOC.read_text(encoding="utf-8")
    assert "--max-replans 200" in text


def test_arm_a_is_the_documented_baseline():
    """Arm A is the no-lever control: an EXTRA that quietly acquired a lever
    would make the A/B compare two levered arms."""
    lines = _extra_lines()
    a = _parse_extra(lines[0])
    assert lines[0].strip() == "", f"arm A's EXTRA must be empty, got {lines[0]!r}"
    assert a.nfe is None and not a.terminal_veto and not a.parity_fixes
