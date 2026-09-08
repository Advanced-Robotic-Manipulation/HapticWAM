#!/usr/bin/env python3
"""Make an unwarped real-versus-sim camera comparison from frozen preview frames."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--previews',type=Path,default=ROOT/'artifacts/isaac_waffles/d435_camera_audit_20260908/previews')
    parser.add_argument('--out',type=Path,default=ROOT/'docs/results/d435_camera_audit_20260908/comparison.png')
    args=parser.parse_args()
    episodes=[('sept04','September 4',args.evidence/'scene_0000.png'),
              ('aug22','August 22',args.evidence/'fit/ep_waffles_1787395928_000/frame_00000.png')]
    variants=['r2_original','rgb_fov_only','rgb_pose_fit']
    titles=['Recorded RGB','Previous camera\n46.6° × 35.8°',
            'RGB prior; same pose\n54.2° × 42.0°',
            'RGB prior + pose fit\n54.2° × 42.0°']
    fig,axes=plt.subplots(2,4,figsize=(19.2,8.1),dpi=130)
    sources=[]
    for row,(episode,label,real) in enumerate(episodes):
        paths=[real]+[args.previews/episode/v/'sim_first.png' for v in variants]
        for col,path in enumerate(paths):
            image=Image.open(path).convert('RGB')
            assert image.size==(640,480),path
            axes[row,col].imshow(image)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col],fontsize=12)
            if col==0:axes[row,col].set_ylabel(label,fontsize=12)
            sources.append({'row':episode,'column':titles[col].replace('\n','; '),'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    fig.suptitle('D435 RGB camera audit — identical scene geometry across simulated columns',fontsize=16)
    fig.text(.5,.025,'RGB optics are provisional, not queried factory calibration. Pose fit uses September 4 robot landmarks; box/layout remain uncalibrated.\nAll panels retain the full 640 × 480 framing. These are initialization previews, not policy trials.',ha='center',fontsize=11)
    fig.subplots_adjust(left=.035,right=.995,top=.86,bottom=.11,wspace=.025,hspace=.03)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,facecolor='white')
    args.out.with_suffix('.sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(args.out)

if __name__=='__main__':main()
