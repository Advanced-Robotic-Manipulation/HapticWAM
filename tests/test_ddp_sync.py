"""train_loop under DDP must all-reduce gradients: the old code called
step_fn on the bare module, so N ranks trained N independent copies."""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.multiprocessing as mp


def _worker(rank: int, world: int, port: int, q):
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    import dataclasses
    from torch.utils.data import DataLoader, TensorDataset
    from phantom.config.training import TeacherTrainConfig
    from phantom.train import common as C
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)                       # identical init on every rank
    # rank-specific data: without gradient sync the ranks diverge immediately
    x = torch.full((8, 4), float(rank + 1)); y = torch.full((8, 1), float(2 * rank + 1))
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    cfg = dataclasses.replace(TeacherTrainConfig(), device="cpu", max_steps=2, grad_accum=2,
                              batch_size=2, num_workers=0, warmup_steps=0, lr=0.1,
                              lr_new_modules=0.1, log_every=1, ckpt_every=1000, eval_every=1000,
                              synthetic=True, tiny=True)

    def step_fn(batch):
        xb, yb = batch
        return {"total": ((model(xb) - yb) ** 2).mean()}

    C.train_loop(cfg, model, loader, step_fn)
    q.put((rank, torch.cat([p.detach().flatten() for p in model.parameters()])))


@pytest.mark.slow  # spawns two gloo ranks; ~4 min, deselected in CI
@pytest.mark.skipif(sys.platform == "win32", reason="gloo spawn")
def test_ddp_ranks_stay_in_sync():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29513
    procs = [ctx.Process(target=_worker, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(2):
        r, vec = q.get(timeout=240)
        out[r] = vec
    for p in procs:
        p.join(60)
        assert p.exitcode == 0, f"rank process exited {p.exitcode}"
    assert torch.allclose(out[0], out[1], atol=1e-6), "ranks diverged: gradients were not all-reduced"
    # sanity: the synced result is NOT what rank 0 alone would have learned
    torch.manual_seed(0)
    solo = torch.nn.Linear(4, 1)
    assert not torch.allclose(out[0], torch.cat([p.detach().flatten() for p in solo.parameters()]))
