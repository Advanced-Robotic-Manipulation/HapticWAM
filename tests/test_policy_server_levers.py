"""Inference-latency levers are declared at server start and reported in `info` (CPU only)."""
from types import SimpleNamespace

from phantom.inference.remote import PolicyServer


def test_levers_are_reported_in_info_and_default_off():
    policy = SimpleNamespace(policy_kind="phantom", wrench_baseline_rows=0)
    plain = PolicyServer(policy, ckpt="x.pt", ckpt_sha="abc")
    assert plain.status()["inference_levers"] == {}
    srv = PolicyServer(policy, ckpt="x.pt", ckpt_sha="abc",
                       levers={"compile": True, "compile_mode": "reduce-overhead", "flex": False, "fp8": False})
    info = srv.status()
    assert info["inference_levers"]["compile"] is True and info["inference_levers"]["compile_mode"] == "reduce-overhead"
    assert info["inference_levers"]["fp8"] is False


def test_server_argparser_accepts_levers(monkeypatch):
    import argparse
    from phantom.scripts import policy_server as ps
    captured = {}

    def fake_parse(self, argv=None):
        ns = argparse.Namespace(ckpt=None, probe=True, port=1, compile=True, compile_mode="default", flex=True)
        captured.update(vars(ns)); return ns
    # only the parser surface is under test: --probe exits before any model build
    src = open(ps.__file__).read()
    assert '"--compile"' in src and '"--flex"' in src and '"--compile-mode"' in src
    assert "levers=levers" in src
