"""ACE fixed packing: deterministic, parameter-free codecs between structured
contact/action data and 16-channel latent frames (Cosmos-Policy value-frame
style). Rectified-flow targets must be fixed functions of the data — a
trainable output embedder would give the denoiser a moving target — so these
packers have NO learnable parameters. The trainable capacity lives in HHT
(input side) and the readout heads.

Channel map of one CONTACT frame (16 x lat_h x lat_w), n_fingers <= 2:
    0-4    finger 0: [d_disp x, y, z | d_fz | mask(+-1)]  (bilinear cpk -> lat)
    5-9    finger 1: same (zero if absent)
    10     event one-hot as 5 horizontal bands (+1 active / -1 inactive)
    11     CoP Gaussian bumps (finger 0 in left half, finger 1 in right half)
    12     slip score tiled per finger half
    13     finger-0 resultant wrench: 6 values in a 2x3 block grid
    14     finger-1 resultant wrench
    15     wrist F/T: 6 values in a 2x3 block grid

ACTION frame: frame k carries actions [k*apf, (k+1)*apf); action a (within
frame) lives on channel a as action_dim vertical strips along W (value tiled
across the strip; unpack = strip mean — the averaging is built-in denoising).

All inputs are expected pre-normalized (norm stats applied by the dataset).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from phantom.config.hardware import HardwareConfig
from phantom.config.model import N_EVENTS
from phantom.model.sequence import SequenceLayout

_MAX_FINGERS = 2
_PER_FINGER_CH = 5   # d_disp(3) + d_fz(1) + mask(1)
_CH_EVENT = 10
CH_EVENT = _CH_EVENT   # public: rf.training_step supervises this band
_CH_COP = 11
_CH_SLIP = 12
_CH_WRENCH0 = 13
_CH_WRENCH1 = 14
_CH_WRIST = 15


def sigma_group_channels(n_fingers: int) -> dict[str, list[int]]:
    """Packed-contact-frame channels per SigmaHead group — lets the
    heteroscedastic NLL supervise each sigma channel against ITS OWN
    residual (the speed governor consumes per-group sigma, so per-group
    calibration must be earned, not averaged)."""
    ch = {"d_disp": [], "d_fz": [], "mask": []}
    for f in range(n_fingers):
        base = f * _PER_FINGER_CH
        ch["d_disp"] += [base, base + 1, base + 2]
        ch["d_fz"].append(base + 3)
        ch["mask"].append(base + 4)
    ch["cop"] = [_CH_COP]
    ch["slip"] = [_CH_SLIP]
    ch["wrench"] = [_CH_WRENCH0, _CH_WRENCH1][:max(1, n_fingers)]
    ch["wrist"] = [_CH_WRIST]
    return ch


@dataclass
class ContactPackage:
    """Structured contact package for T_c future steps (pipeline.md §2/§4).

    d_disp / d_fz are DELTAS vs. the previous step (step 0 vs. observed t),
    at the cpk resolution from the hardware config.
    """
    event: torch.Tensor        # (B, Tc) long targets  OR (B, Tc, E) probs
    d_disp: torch.Tensor       # (B, Tc, F, 3, cph, cpw)
    d_fz: torch.Tensor         # (B, Tc, F, cph, cpw)
    mask: torch.Tensor         # (B, Tc, F, cph, cpw) in [0, 1]
    cop: torch.Tensor          # (B, Tc, F, 2) in [-1, 1], NaN = undefined
    slip: torch.Tensor         # (B, Tc, F)
    wrench: torch.Tensor       # (B, Tc, F, 6)
    wrist: torch.Tensor        # (B, Tc, 6)

    @property
    def batch(self) -> int:
        return self.d_disp.shape[0]

    @property
    def horizon(self) -> int:
        return self.d_disp.shape[1]

    @property
    def n_fingers(self) -> int:
        return self.d_disp.shape[2]

    def to(self, *args, **kw) -> "ContactPackage":
        return ContactPackage(**{f.name: getattr(self, f.name).to(*args, **kw)
                                 for f in fields(self)})

    def detach(self) -> "ContactPackage":
        return ContactPackage(**{f.name: getattr(self, f.name).detach()
                                 for f in fields(self)})

    def event_probs(self) -> torch.Tensor:
        """(B, Tc, E) — one-hot if `event` holds class ids."""
        if self.event.dtype in (torch.long, torch.int64, torch.int32):
            return F.one_hot(self.event.long(), N_EVENTS).float()
        return self.event


def _blocks(n: int, length: int) -> list[slice]:
    """Deterministic partition of [0, length) into n contiguous slices."""
    edges = torch.linspace(0, length, n + 1).round().long().tolist()
    return [slice(edges[i], edges[i + 1]) for i in range(n)]


class ContactPacker:
    def __init__(self, hw: HardwareConfig, layout: SequenceLayout):
        assert hw.n_fingers <= _MAX_FINGERS, (
            f"ContactPacker channel map supports at most {_MAX_FINGERS} fingers, "
            f"got {hw.n_fingers} (extend the map or drop sensors)")
        self.hw = hw
        self.layout = layout
        self.cph, self.cpw = hw.cpk_shape
        self.H, self.W = layout.lat_h, layout.lat_w
        self._h_halves = _blocks(max(hw.n_fingers, 1), self.W)
        self._event_bands = _blocks(N_EVENTS, self.H)
        self._wr_rows = _blocks(2, self.H)
        self._wr_cols = _blocks(3, self.W)
        # CoP bump grid
        ys = torch.linspace(-1, 1, self.H)
        self._cop_yy = ys.view(-1, 1)

    # ------------------------------------------------------------------
    def pack(self, cpk: ContactPackage) -> torch.Tensor:
        """-> (B, lat_c, Tc, lat_h, lat_w)"""
        B, Tc, Fn = cpk.d_disp.shape[:3]
        H, W = self.H, self.W
        dev, dt = cpk.d_disp.device, torch.float32
        x = torch.zeros(B, self.layout.lat_c, Tc, H, W, device=dev, dtype=dt)

        def up(field: torch.Tensor) -> torch.Tensor:
            # (B*Tc*?, c, cph, cpw) -> bilinear (H, W)
            return F.interpolate(field, size=(H, W), mode="bilinear", align_corners=False)

        for f in range(Fn):
            base = f * _PER_FINGER_CH
            disp = up(cpk.d_disp[:, :, f].reshape(B * Tc, 3, self.cph, self.cpw))
            x[:, base:base + 3] = disp.reshape(B, Tc, 3, H, W).permute(0, 2, 1, 3, 4)
            dfz = up(cpk.d_fz[:, :, f].reshape(B * Tc, 1, self.cph, self.cpw))
            x[:, base + 3] = dfz.reshape(B, Tc, H, W)
            m = up(cpk.mask[:, :, f].reshape(B * Tc, 1, self.cph, self.cpw))
            x[:, base + 4] = m.reshape(B, Tc, H, W) * 2.0 - 1.0

        # event bands
        probs = cpk.event_probs().to(dt)                     # (B, Tc, E)
        for e, band in enumerate(self._event_bands):
            x[:, _CH_EVENT, :, band, :] = (probs[:, :, e] * 2.0 - 1.0)[..., None, None]

        # CoP bumps: each finger's half of the channel is its own [-1,1]^2
        # canvas (local coordinates), so both fingers' full CoP range fits
        yy = self._cop_yy.to(dev)                            # (H, 1)
        for f in range(Fn):
            half = self._h_halves[f]
            cop = cpk.cop[:, :, f]                           # (B, Tc, 2) in [-1,1]
            cop_valid = torch.nan_to_num(cop, nan=0.0)
            amp = (~torch.isnan(cop[..., 0])).to(dt)         # 0 where undefined
            n_half = half.stop - half.start
            xs = torch.linspace(-1, 1, n_half, device=dev)   # LOCAL x coords
            r2 = ((xs.view(1, 1, 1, -1) - cop_valid[..., 0, None, None]) ** 2
                  + (yy.view(1, 1, -1, 1) - cop_valid[..., 1, None, None]) ** 2)
            bump = torch.exp(-r2 / (2 * 0.25 ** 2)) * amp[..., None, None]
            x[:, _CH_COP, :, :, half] = bump

        # slip tiles
        for f in range(Fn):
            x[:, _CH_SLIP, :, :, self._h_halves[f]] = cpk.slip[:, :, f, None, None]

        # wrench / wrist block grids
        for f, ch in zip(range(Fn), (_CH_WRENCH0, _CH_WRENCH1)):
            self._tile6(x, ch, cpk.wrench[:, :, f])
        self._tile6(x, _CH_WRIST, cpk.wrist)
        return x

    def _tile6(self, x: torch.Tensor, ch: int, vals: torch.Tensor) -> None:
        """vals (B, Tc, 6) -> 2x3 block grid on channel ch."""
        k = 0
        for r in self._wr_rows:
            for c in self._wr_cols:
                x[:, ch, :, r, c] = vals[..., k, None, None]
                k += 1

    # ------------------------------------------------------------------
    def unpack(self, x: torch.Tensor) -> ContactPackage:
        """(B, lat_c, Tc, lat_h, lat_w) -> ContactPackage (event as probs)."""
        B, _, Tc, H, W = x.shape
        Fn = self.hw.n_fingers

        def down(field: torch.Tensor, c: int) -> torch.Tensor:
            return F.interpolate(field.reshape(B * Tc, c, H, W),
                                 size=(self.cph, self.cpw), mode="bilinear",
                                 align_corners=False).reshape(B, Tc, c, self.cph, self.cpw)

        d_disp, d_fz, mask, cop, slip, wrench = [], [], [], [], [], []
        xs_full = torch.linspace(-1, 1, W, device=x.device)
        ys_full = torch.linspace(-1, 1, H, device=x.device)
        for f in range(Fn):
            base = f * _PER_FINGER_CH
            d_disp.append(down(x[:, base:base + 3].permute(0, 2, 1, 3, 4), 3))
            d_fz.append(down(x[:, base + 3].unsqueeze(2), 1)[:, :, 0])
            m = down((x[:, base + 4] + 1.0).mul(0.5).clamp(0, 1).unsqueeze(2), 1)[:, :, 0]
            mask.append(m)
            # CoP: bump-weighted centroid in the half's LOCAL coordinates
            half = self._h_halves[f]
            bump = x[:, _CH_COP, :, :, half].clamp_min(0)    # (B, Tc, H, w_half)
            n_half = half.stop - half.start
            xs = torch.linspace(-1, 1, n_half, device=x.device)
            total = bump.sum((-2, -1)).clamp_min(1e-6)
            cx = (bump * xs.view(1, 1, 1, -1)).sum((-2, -1)) / total
            cy = (bump * ys_full.view(1, 1, -1, 1)).sum((-2, -1)) / total
            cop.append(torch.stack([cx, cy], dim=-1))
            slip.append(x[:, _CH_SLIP, :, :, half].mean((-2, -1)))
            wrench.append(self._untile6(x, _CH_WRENCH0 if f == 0 else _CH_WRENCH1))

        event_logits = torch.stack(
            [x[:, _CH_EVENT, :, band, :].mean((-2, -1)) for band in self._event_bands],
            dim=-1)                                          # (B, Tc, E) in ~[-1, 1]
        event_probs = ((event_logits + 1.0) * 0.5).clamp(1e-4, 1.0)
        event_probs = event_probs / event_probs.sum(-1, keepdim=True)

        return ContactPackage(
            event=event_probs,
            d_disp=torch.stack(d_disp, dim=2),
            d_fz=torch.stack(d_fz, dim=2),
            mask=torch.stack(mask, dim=2),
            cop=torch.stack(cop, dim=2),
            slip=torch.stack(slip, dim=2),
            wrench=torch.stack(wrench, dim=2),
            wrist=self._untile6(x, _CH_WRIST),
        )

    def _untile6(self, x: torch.Tensor, ch: int) -> torch.Tensor:
        vals = []
        for r in self._wr_rows:
            for c in self._wr_cols:
                vals.append(x[:, ch, :, r, c].mean((-2, -1)))
        return torch.stack(vals, dim=-1)

    # ------------------------------------------------------------------
    def flatten_summary(self, cpk: ContactPackage, step: int = 0) -> torch.Tensor:
        """Deterministic summary vector of one future step for ACC's cpk_mlp:
        [event probs (E)] + per finger [dfz mean, dfz absmax, slip, cop(2),
        wrench(6)] + wrist(6).  -> (B, summary_dim)"""
        s = min(step, cpk.horizon - 1)
        parts = [cpk.event_probs()[:, s]]
        for f in range(cpk.n_fingers):
            dfz = cpk.d_fz[:, s, f]
            parts.append(dfz.mean((-2, -1), keepdim=False).unsqueeze(-1))
            parts.append(dfz.abs().amax((-2, -1)).unsqueeze(-1))
            parts.append(cpk.slip[:, s, f].unsqueeze(-1))
            parts.append(torch.nan_to_num(cpk.cop[:, s, f], nan=0.0))
            parts.append(cpk.wrench[:, s, f])
        parts.append(cpk.wrist[:, s])
        return torch.cat(parts, dim=-1)

    def summary_dim(self) -> int:
        return N_EVENTS + self.hw.n_fingers * (1 + 1 + 1 + 2 + 6) + 6


class ActionPacker:
    def __init__(self, hw: HardwareConfig, layout: SequenceLayout):
        self.hw = hw
        self.layout = layout
        self.A = hw.control.action_dim
        self.apf = layout.actions_per_frame
        assert self.apf <= layout.lat_c, "actions per frame exceed latent channels"
        self._strips = _blocks(self.A, layout.lat_w)

    def pack(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, H, A) -> (B, lat_c, Ta, lat_h, lat_w)"""
        B, Hz, A = actions.shape
        assert A == self.A and Hz == self.hw.control.chunk_horizon, \
            f"action chunk shape {(Hz, A)} != ({self.hw.control.chunk_horizon}, {self.A})"
        Ta = Hz // self.apf
        x = torch.zeros(B, self.layout.lat_c, Ta, self.layout.lat_h, self.layout.lat_w,
                        device=actions.device, dtype=torch.float32)
        a_frames = actions.reshape(B, Ta, self.apf, A)
        for a in range(self.apf):
            for j, strip in enumerate(self._strips):
                x[:, a, :, :, strip] = a_frames[:, :, a, j, None, None]
        return x

    def unpack(self, x: torch.Tensor) -> torch.Tensor:
        """(B, lat_c, Ta, lat_h, lat_w) -> (B, H, A); strip means = built-in denoising."""
        B, _, Ta = x.shape[:3]
        out = torch.zeros(B, Ta, self.apf, self.A, device=x.device, dtype=torch.float32)
        for a in range(self.apf):
            for j, strip in enumerate(self._strips):
                out[:, :, a, j] = x[:, a, :, :, strip].mean((-2, -1))
        return out.reshape(B, Ta * self.apf, self.A)
