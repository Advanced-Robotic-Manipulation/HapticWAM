"""Does information that the structural mask is supposed to hide from ACTION queries
(VIDEO_GEN latents; the intent chunk via the VIDEO_GEN-only AdaLN) reach the ACTION-frame
velocity output anyway (2-hop via OBS/VIDEO_COND tokens, which are allowed to attend VIDEO_GEN)?
Tiny random-init teacher; ACC lambdas are 0 at init so the ACC bias path is OFF."""
import sys, numpy as np, torch
sys.path.insert(0, "/Users/sannikov/GitHub/phantom/tests")
from phantom_test_utils import make_hw
from phantom.config.paths import load_paths
from phantom.train.builder import build_model
from phantom.model.sequence import FrameGroup, SequenceLayout
torch.manual_seed(0)
hw = make_hw(tactile={"field": {"h": 48, "w": 64}, "raw_img": {"h": 60, "w": 80, "c": 1},
                      "infer_img": {"h": 60, "w": 80, "c": 1}, "rate_hz": 30.0},
             cameras={"scene": {"color": {"h": 60, "w": 80, "c": 3}, "fps": 10.0}},
             recording={"field_ds": {"h": 24, "w": 32}, "field_ds_rate_hz": 30.0, "keyframe_rate_hz": 5.0,
                        "keyframe_ds": {"h": 24, "w": 32}, "infer_img_rate_hz": 10.0, "zarr_chunk_frames": 16},
             derived={"cpk_downsample": 4}, wrist_ft={"window_s": 0.1})
pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
net, layout = pm.rf.net, pm.layout
assert float(net.phantom_bias_lambdas.abs().sum()) == 0.0
B, T = 1, layout.t_total
g = torch.Generator().manual_seed(1)
x = torch.randn(B, layout.lat_c, T, layout.lat_h, layout.lat_w, generator=g)
cond = torch.from_numpy(layout.cond_mask_T()).reshape(1, 1, T, 1, 1).expand(B, 1, T, layout.lat_h, layout.lat_w).float()
tBT = torch.full((B, T), 500.0); tBT[:, torch.from_numpy(layout.cond_mask_T())] = 0.0
ctx = torch.randn(B, pm.bb.text_emb_seq_len, pm.bb.text_emb_dim, generator=g)
intent = torch.randn(B, hw.control.chunk_horizon, hw.control.action_dim, generator=g)
def act_v(x_, intent_, layout_=layout):
    with torch.no_grad():
        out = net(x_B_C_T_H_W=x_, timesteps_B_T=tBT[:, :layout_.t_total], crossattn_emb=ctx,
                  condition_video_input_mask_B_C_T_H_W=cond[:, :, :layout_.t_total], action=intent_, acc_inputs=None, layout=layout_)
    return out.velocity_B_C_T_H_W[:, :, layout_.frame_slice(FrameGroup.ACTION)].float()
base = act_v(x, intent)
def rel(a, b): return float((a - b).norm() / (b.norm() + 1e-9))
# (1) intent chunk perturbed (reaches only the VIDEO_GEN AdaLN by construction)
print("intent chunk randomised          -> ACTION velocity rel change %.3f" % rel(act_v(x, intent + torch.randn_like(intent)), base))
# (2) VIDEO_GEN latent input randomised (ACTION queries are masked from these keys)
x2 = x.clone(); x2[:, :, layout.frame_slice(FrameGroup.VIDEO_GEN)] = torch.randn_like(x2[:, :, layout.frame_slice(FrameGroup.VIDEO_GEN)])
print("VIDEO_GEN latents randomised     -> ACTION velocity rel change %.3f" % rel(act_v(x2, intent), base))
# (3) control: an ACTION-slot noise change (must change) and a CONTACT-slot change
x3 = x.clone(); x3[:, :, layout.frame_slice(FrameGroup.OBS_PROPRIO)] = torch.randn_like(x3[:, :, layout.frame_slice(FrameGroup.OBS_PROPRIO)])
print("OBS_PROPRIO frame randomised     -> ACTION velocity rel change %.3f" % rel(act_v(x3, intent), base))
# (4) drop_video layout: same x minus the VIDEO_GEN frames
ld = SequenceLayout.build(pm.bb, pm.mc, hw, student=False, drop_video=True)
keep = [t for t in range(T) if t not in range(*layout.frame_slice(FrameGroup.VIDEO_GEN).indices(T))]
xd = x[:, :, keep]
print("drop_video=True at inference     -> ACTION velocity rel change %.3f" % rel(act_v(xd, intent, ld), base))
print("(structural mask per sequence.py:190-202 blocks only CONTACT/ACTION -> VIDEO_GEN; OBS_*/VIDEO_COND queries still read VIDEO_GEN keys)")

# CONTROL: extend the structural mask so that EVERY non-VIDEO_GEN query is blocked from VIDEO_GEN keys.
import phantom.model.attention_bias as ab
def full_block(layout_):
    bias = np.zeros((layout_.t_total, layout_.t_total), dtype=np.float32)
    if layout_.has(FrameGroup.VIDEO_GEN):
        vid = layout_.frame_slice(FrameGroup.VIDEO_GEN)
        for s in layout_.slots:
            if s.group != FrameGroup.VIDEO_GEN:
                bias[s.t_slice, vid] = -1e4
    return bias
SequenceLayout.structural_attn_bias = full_block
net._structural_cache.clear()
base2 = act_v(x, intent)
print("CONTROL (all non-video queries blocked from VIDEO_GEN): intent randomised -> %.6f ; VIDEO_GEN randomised -> %.6f ; drop_video -> %.6f"
      % (rel(act_v(x, intent + torch.randn_like(intent)), base2), rel(act_v(x2, intent), base2), rel(act_v(xd, intent, ld), base2)))
