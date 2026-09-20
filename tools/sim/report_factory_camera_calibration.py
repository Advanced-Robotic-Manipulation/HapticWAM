#!/usr/bin/env python3
"""Create full-frame comparisons and verify frozen native Isaac previews."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from fit_d435_camera_audit import project

ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--previews',type=Path,default=ROOT/'artifacts/isaac_waffles/d435_factory_calibration_20260908')
    p.add_argument('--out',type=Path,default=ROOT/'tests/fixtures/reference/d435_factory_calibration_20260908')
    a=p.parse_args();d=a.out
    reg=json.loads((d/'registration.json').read_text());g=json.loads((d/'grid_fit.json').read_text())
    cfg_path=ROOT/'configs/sim/waffles_d435_factory_20260908_r4.json';cfg=json.loads(cfg_path.read_text())
    # First-row comparison retains exact historical camera/geometry output.
    old=ROOT/'artifacts/isaac_waffles/d435_camera_audit_20260908/previews/sept04/r2_original/sim_first.png'
    paths=[a.evidence/'scene_0000.png',old,a.previews/'final_sept04/sim_first.png']
    titles=['Recorded D435 RGB','Previous estimated camera / layout\n46.61° × 35.81°','Factory K + measured geometry\n55.38° × 43.06°']
    fig,axes=plt.subplots(1,3,figsize=(15.6,4.9),dpi=135)
    sources=[]
    for ax,path,title in zip(axes,paths,titles):
        img=Image.open(path);assert img.size==(640,480)
        ax.imshow(img);ax.set(xticks=[],yticks=[],title=title)
        sources.append({'title':title,'path':str(path),'sha256':sha(path)})
    fig.suptitle('D435 camera and waffle-rig reconstruction — full 640 × 480 frames',fontsize=14)
    fig.text(.5,.025,'Right: fixed factory intrinsics, metric table pose, measured packet/bin sizes. Materials and gripper remain approximations.\nInitialization preview only; no teacher trial or task-success claim.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.01,right=.995,top=.81,bottom=.14,wspace=.018);fig.savefig(d/'comparison.png',facecolor='white');plt.close(fig)
    # Exact grid projections, with no image warp.
    transform=np.asarray(reg['world_from_grid']);points=np.asarray(g['assignment']['object_xyz_m'])
    predicted=project(cfg['camera'],points@transform[:3,:3].T+transform[:3,3]);observed=np.asarray(g['assignment']['observed_pixels'])
    fig,ax=plt.subplots(figsize=(12.8,9.6),dpi=100);ax.imshow(Image.open(paths[0]))
    ax.scatter(observed[:,0],observed[:,1],s=40,edgecolors='lime',facecolors='none',linewidths=.8,label='Observed table holes')
    ax.scatter(predicted[:,0],predicted[:,1],s=5,color='magenta',label='Projected 50 mm lattice')
    for label,key in [('packet','packet_top'),('opening','bin_opening')]:
        for suffix,color,style in [('observations','lime','-'),('predictions','magenta','--')]:
            xy=np.array(reg[f'{key}_{suffix}_px']);xy=np.r_[xy,xy[:1]]
            ax.plot(xy[:,0],xy[:,1],color=color,ls=style,lw=1.4,label=f'{suffix}: {label}')
    ax.set(xlim=(-.5,639.5),ylim=(479.5,-.5),xticks=[],yticks=[],title='Factory D435 K + measured geometry, projected onto original RGB\nTable 0.99 px RMS | packet 1.36 px | box opening 3.58 px')
    ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(d/'projection_overlay.png');plt.close(fig)
    h=json.loads((d/'heldout_annotations.json').read_text());fig,axes=plt.subplots(3,3,figsize=(15,11.8),dpi=115)
    for row,(ep,res) in enumerate(zip(h['episodes'],reg['heldout_episodes'])):
        for col,(f,v) in enumerate(zip(ep['frames'],res['landmarks'])):
            ax=axes[row,col];ax.imshow(Image.open(a.evidence/'heldout'/ep['episode']/f['frame_file']))
            o,vp=np.array(v['observed_px']),np.array(v['predicted_px'])
            ax.scatter(*o,s=110,edgecolors='lime',facecolors='none',linewidths=1.5);ax.scatter(*vp,s=55,marker='+',color='magenta')
            ax.plot([o[0],vp[0]],[o[1],vp[1]],'magenta',lw=1)
            ax.set(xticks=[],yticks=[],title=f'{ep["date"]}  {f["frame_file"]} | {v["error_px"]:.1f} px')
    fig.suptitle('Same camera across three held-out episodes and nine measured arm poses\nGreen: independent wrist label; magenta: nominal CAD projection. No held-out labels used for fitting.',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(d/'heldout_projection.png');plt.close(fig)
    # Two complete native previews with distinct measured initial arm states.
    report={'config_sha256':sha(cfg_path),'scope':'Native build/initialization previews; no policy or trajectory-controller qualification','previews':[]}
    fig,axes=plt.subplots(2,2,figsize=(10.6,9.5),dpi=135)
    for row,(name,real) in enumerate([
        ('final_sept04',a.evidence/'scene_0000.png'),
        ('final_sept01_heldout',a.evidence/'heldout/ep_teacher_waffles_1788262056_000/frame_00000.png')]):
        folder=a.previews/name;effective=json.loads((folder/'effective_config.json').read_text());run=json.loads((folder/'run.json').read_text());lens=json.loads((folder/'camera_projection.json').read_text())
        assert effective==cfg,'Native effective config differs from final config'
        assert run['frames']==4 and run['mode']=='replay' and lens['readback_matches_config']
        np.testing.assert_array_equal(lens['intrinsics_px'],[[cfg['camera']['fx'],0,cfg['camera']['cx']],[0,cfg['camera']['fy'],cfg['camera']['cy']],[0,0,1]])
        subprocess.run(['ffmpeg','-v','error','-i',str(folder/'sim.mp4'),'-f','null','-'],check=True,capture_output=True)
        trace=np.load(folder/'sim_trace.npz');assert len(trace['t'])==4
        entry={'name':name,'mode':run['mode'],'frames':run['frames'],'duration_s':run['duration_s'],'full_video_decode_passed':True,'exact_factory_K_readback':True,'effective_config_matches_final':True,'artifacts':{f:sha(folder/f) for f in ['sim_first.png','sim.mp4','run.json','camera_projection.json','effective_config.json','sim_trace.npz']}}
        report['previews'].append(entry)
        for col,path in enumerate([real,folder/'sim_first.png']):
            axes[row,col].imshow(Image.open(path));axes[row,col].set(xticks=[],yticks=[],title=('Recorded RGB' if col==0 else 'Isaac: same camera and scene')+('\nSeptember 4 fit episode' if row==0 else '\nSeptember 1 held-out episode'))
    fig.suptitle('Native Isaac previews with two measured arm starts',fontsize=14);fig.subplots_adjust(left=.015,right=.995,top=.89,bottom=.015,hspace=.22,wspace=.035);fig.savefig(d/'native_comparison.png');plt.close(fig)
    # Independent real-to-rendered grid centroid check. Use matching lattice IDs
    # from mathematical projections; reject >4px assignments and duplicate IDs.
    gray=cv2.cvtColor(cv2.imread(str(a.previews/'final_sept04/sim_first.png')),cv2.COLOR_BGR2GRAY)
    _,_,stats,centers=cv2.connectedComponentsWithStats((gray<60).astype('uint8'))
    centers=np.array([c for s,c in zip(stats,centers) if 2<=s[4]<=150 and 1<=s[2]<=20 and 1<=s[3]<=20])
    distance=np.linalg.norm(predicted[:,None]-centers[None],axis=2);index=distance.argmin(1);keep=distance.min(1)<4
    assert len(set(index[keep]))==int(keep.sum()),'Duplicate rendered hole assignment'
    error=np.linalg.norm(centers[index[keep]]-observed[keep],axis=1)
    report['rendered_grid_alignment']={'correspondences':int(keep.sum()),'available_real_holes':len(observed),'real_vs_rendered_centroid_rmse_px':float(np.sqrt(np.mean(error**2))),'max_error_px':float(error.max()),'matching_gate_px':4,'dark_threshold':60,'scope':'Selected visible table region; thresholded dark centers, not photometric or whole-image fidelity'}
    report['source_hashes']={str(path):sha(path) for path in [Path(__file__),cfg_path,d/'registration.json',d/'grid_fit.json']}
    (d/'runtime_verification.json').write_text(json.dumps(report,indent=2)+'\n');(d/'comparison.sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(report['rendered_grid_alignment'],indent=2));print(d/'comparison.png')

if __name__=='__main__':main()
