import subprocess
import argparse
import sys
import os
import logging
import paho.mqtt.client as mqtt


DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from common import loadConfig
from inspector import sendNodeStateMsg


def main(args):
    nodename = args.nodename

    cfg_file = f"{ROOTDIR}/config"
    cfg = loadConfig(cfg_file)
    broker = cfg['Project']['mqtt_broker']
    client = mqtt.Client()
    client.connect(broker)

    logging.debug("{} started.".format(nodename))
    sendNodeStateMsg(client, nodename, "started")

    video_list = []

    if args.run_folder != None:
        for videoname in os.listdir(args.run_folder):
            if videoname.endswith('.avi'):
                video_list.append(os.path.join(args.run_folder, videoname))

    if args.run != None:
        video_list.append(args.run)

    cfg_file = f"{ROOTDIR}/config"
    cfg = loadConfig(cfg_file)
    for i in range(len(video_list)):
        for node_name, node_info in cfg.items():
            if 'file_name' in node_info:
                if node_info['file_name'] in video_list[i]:
                    csv_file = os.path.join(os.path.dirname(video_list[i]), f"{node_name}.csv")
                    break
        subprocess.call(f"python3 FlatballActionAnalyzer.py --input_video {video_list[i]} --input_csv {csv_file} \
                            --output_folder {args.output_folder}", shell=True)
        #subprocess.call(f"python3 VibePklParser.py ./output/CameraReaderL_482.avi/vibe_output.pkl ./input/28124278.cfg ./", shell=True)

    logging.debug("{} terminated.".format(nodename))
    sendNodeStateMsg(client, nodename, "terminated")


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

    parser.add_argument('--nodename', type=str, default = 'Analyzer', help = 'mqtt node name (default: RnnPredictor)')

    args = parser.parse_args()
    main(args)