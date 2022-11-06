import subprocess
import argparse
import os
import sys
import logging

from Backend_High_VibePklParser import Backend_High_VibePklParser

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from common import loadConfig
from inspector import sendNodeStateMsg

def main(args):

    video_list = []

    if args.run_folder != None:
        for videoname in os.listdir(args.run_folder):
            if videoname.endswith('L.avi'):
                video_list.append(os.path.join(args.run_folder, videoname))

    if args.run != None:
        video_list.append(args.run)

    #logging.warning(video_list)
    video_list = sorted(video_list)
    if not args.no_vibe:
        for i in range(len(video_list)):
            subprocess.call(f"python3 ../lib/PklGenerator.py --vid_file={video_list[i]} --output_folder={args.output_folder} --no_render", shell=True)

    pkl_list = []
    for folder in os.listdir(args.output_folder):
        folder_path = os.path.join(args.output_folder, folder)
        if os.path.isdir(folder_path) and os.path.isfile(os.path.join(folder_path, "vibe_output.pkl")):
            pkl_list.append(os.path.join(folder_path, "vibe_output.pkl"))
    pkl_list = sorted(pkl_list)
    logging.warning(pkl_list)

    strike_fid = int(int(args.fps)*1.5)
    for i in range(len(pkl_list)):
        logging.warning(f'vibe_pkl: {pkl_list[i]}\nsource_video: {video_list[i]}')
        subprocess.call(f"python3 Backend_High_VibePklParser.py {pkl_list[i]} {args.camera_cfg} {video_list[i]}", shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('--camera_cfg', '-c', type=str, required=True,
                        help='camera cfg VibePklParser need')

    parser.add_argument('--output_folder', '-o', type=str, required=True,
                        help='output folder')

    action.add_argument('--run', type=str,
                        help='input videoname then run one video')

    action.add_argument('--run_folder', '-f', type=str,
                        help='input folder name then run all videos in folder (pkl, camera cfg, output folder)')

    parser.add_argument('--fps', type=str,
                        help='source video fps')

    parser.add_argument('--no_vibe', action='store_true',
                    help='not to run vibe, just display')

    args = parser.parse_args()
    main(args)
