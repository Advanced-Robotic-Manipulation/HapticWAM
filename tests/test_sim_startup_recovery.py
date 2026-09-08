"""Fail closed before replacing a startup-only experiment attempt."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).parents[1] / "docs/results/teacher_scene_v10_20260908/recover_startup_timeout.py"
spec = importlib.util.spec_from_file_location("startup_recovery", MODULE)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class StartupRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        source = self.base / "source"
        self.runner = source / "docs/results/teacher_scene_v10_20260908/run_study.py"
        self.runner.parent.mkdir(parents=True)
        self.runner.write_text("# frozen runner\n")
        self.output = self.base / "campaign"
        study = {"runtime": {"source": str(source), "output": str(self.output), "port": 0}}
        write(self.base / "study.json", study)
        sha = recovery.digest(self.base / "study.json")
        self.addCleanup(patch.stopall)
        patch.object(recovery, "EXPECTED_STUDY", sha).start()
        patch.object(recovery, "EXPECTED_RUNNER", recovery.digest(self.runner)).start()
        write(self.output / "frozen_inputs.json", {"runner_sha256": recovery.EXPECTED_RUNNER, "study_sha256": sha})
        rows = {}
        for i in range(27):
            key = f"old_bounded/trial{i}"
            rows[key] = {"status": "completed"}
            d = self.output / "old_bounded/rollouts" / f"trial{i}"
            write(d / "run_status.json", {"status": "completed"})
            write(d / "runtime_audit.json", {"valid": True})
            (d / "raw.bin").write_bytes(bytes([i]) * 20)
        rows[recovery.TRIAL] = {"status": "timeout"}
        write(self.output / "progress.json", {"status": "error", "active_trial": recovery.TRIAL, "trials": rows})
        self.d = self.output / "r4_legacy/rollouts" / recovery.TRIAL.split("/")[1]
        config = self.base / "scene.json"
        write(config, {"measured_scene": True})
        command = ["launch_waffles.sh", "--output", str(self.d), "--config", str(config)]
        write(self.d / "run_status.json", {"status": "timeout", "exit_code": -15, "started_unix_s": 1788870582.8,
                                          "finished_unix_s": 1788871482.8, "command": command})
        self.server = self.output / "servers/server1"
        write(self.server / "server.json", {"teacher": True})
        (self.server / "server.log").write_text("2026-09-08 15:28:06,636 phantom.inference.remote: client #26 connected\n")
        write(self.d / "case.json", {"study_sha256": sha, "schedule_index": 27, "attempt": 1,
                                    "server_metadata": str(self.server / "server.json")})
        write(self.d / "command.json", command)
        write(self.d / "effective_config.json", {"measured_scene": True})
        write(self.d / "resources_before.json", {"gpu": "free"})
        write(self.d / "server_ready.json", {"teacher": True})
        (self.d / "isaac.log").touch()
        for name in ["plan.json", "startup_resources.json"]:
            write(self.output / name, {})
        (self.base / "execute_after_gpu_free.log").write_text("TimeoutExpired after 900 seconds\n")

    def test_explicit_recovery_preserves_bytes_and_completed_trials(self):
        original = recovery.hashes(self.d)
        preview = recovery.recover(self.base)
        self.assertTrue(self.d.exists())
        self.assertFalse((self.base / "amendments").exists())
        result = recovery.recover(self.base, execute=True)
        self.assertEqual(result["completed_trials_preserved"], 27)
        self.assertEqual(result["attempt_file_sha256"], original)
        self.assertEqual(result["completed_trial_file_sha256"], preview["completed_trial_file_sha256"])
        self.assertEqual(recovery.hashes(Path(result["archive"]) / "attempt1"), original)
        self.assertFalse(self.d.exists())
        self.assertEqual(recovery.digest(self.runner), recovery.EXPECTED_RUNNER)
        self.assertEqual(json.loads((self.base / "amendments/startup_timeout_recovery.json").read_text()), result)

    def test_physical_evidence_cannot_be_retried(self):
        (self.d / "execution_trace.jsonl").write_text('{"command": 1}\n')
        with self.assertRaisesRegex(ValueError, "physical rollout evidence"):
            recovery.recover(self.base, execute=True)
        self.assertFalse((self.base / "amendments").exists())

    def test_nonempty_startup_log_requires_new_review(self):
        (self.d / "isaac.log").write_text("app started\n")
        with self.assertRaisesRegex(ValueError, "not empty"):
            recovery.recover(self.base, execute=True)

    def test_policy_connection_disallows_startup_only_recovery(self):
        (self.server / "server.log").write_text("2026-09-08 15:30:06,636 phantom.inference.remote: client #27 connected\n")
        with self.assertRaisesRegex(ValueError, "policy client connected"):
            recovery.recover(self.base, execute=True)

    def test_changed_runner_disallows_recovery(self):
        self.runner.write_text("# altered runner\n")
        with self.assertRaisesRegex(ValueError, "runner changed"):
            recovery.recover(self.base, execute=True)


if __name__ == "__main__":
    unittest.main()
