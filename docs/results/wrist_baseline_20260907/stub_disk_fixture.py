"""Test-only disk fixture for five existing fully stubbed run_deploy tests.

No production configuration or disk guard is changed. The checkout's filesystem
has less than the native launcher's 5 GB minimum; these tests do no recording and
replace DeploymentRuntime/drivers/policy before calling main().
"""

import pytest

STUBBED_MAIN_TESTS = {
    "test_run_deploy_passes_a_35_s_episode_budget_by_default",
    "test_the_zfloor_tag_is_honest_when_the_floor_is_off",
    "test_run_deploy_refuses_to_start_on_an_inverted_envelope",
    "test_the_homing_jitter_is_seeded_from_the_episode_seed",
    "test_the_realised_start_pose_is_tagged",
}


@pytest.fixture(autouse=True)
def disk_for_stubbed_main_only(monkeypatch, request):
    if request.node.name in STUBBED_MAIN_TESTS:
        from phantom.scripts import run_deploy

        monkeypatch.setattr(run_deploy, "preflight_disk", lambda _out: 0)
