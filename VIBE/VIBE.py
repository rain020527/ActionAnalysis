# LIB for running VIBE
import platform

if platform.system() == 'Linux':
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # for Linux GPU-accelerated rendering

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from os import makedirs, getcwd, listdir
from os.path import isfile, join, basename, isabs, splitext, realpath
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
from vibe_lib.models.vibe import VIBE_Demo
from vibe_lib.utils.renderer import Renderer
from vibe_lib.dataset.inference import Inference
from vibe_lib.utils.smooth_pose import smooth_pose
from vibe_lib.data_utils.kp_utils import convert_kps
from vibe_lib.utils.pose_tracker import run_posetracker
from vibe_lib.utils.demo_utils import (
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

# LIB for this VIBE node
import threading
from os import sys, chdir
from os.path import dirname, realpath, isabs
import logging
import argparse
import json
import paho.mqtt.client as mqtt

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from common import load_config, insertById
from inspector import sendPerformance, sendNodeStateMsg

MIN_NUM_FRAMES = 25

def vibe_3d_pose(args, model, mqttObj=None, topic='vibe'):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    video_file = args['vid_file']

    if not isfile(video_file):
        logging.error('input video does not exist. video path = {}'.format(video_file))
        return

    output_path = join(args['output_folder'], splitext(basename(video_file))[0])
    makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    logging.info('Input video number of frames {}'.format(num_frames))
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    if args['tracking_method'] == 'pose':
        if not isabs(video_file):
            video_file = join(getcwd(), video_file)
        tracking_results = run_posetracker(video_file, staf_folder=args['staf_dir'], display=args['display'])
    else:
        # run multi object tracker
        mot = MPT(device=device, batch_size=args['tracker_batch_size'], display=args['display'], detector_type=args['detector'],
            output_format='dict', yolo_img_size=args['yolo_img_size'])
        tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Run VIBE on each person ========= #
    logging.info('Running VIBE on each tracklet ...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if args['tracking_method'] == 'bbox':
            bboxes = tracking_results[person_id]['bbox']
        elif args['tracking_method'] == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']
        dataset = Inference(image_folder=image_folder, frames=frames, bboxes=bboxes, joints2d=joints2d, scale=bbox_scale)
        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False
        #dataloader = DataLoader(dataset, batch_size=args['vibe_batch_size'], num_workers=16)
        dataloader = DataLoader(dataset, batch_size=args['vibe_batch_size'], num_workers=2)

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)

            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args['run_smplify'] and args['tracking_method'] == 'pose':
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            logging.info('Update ratio after Temporal SMPLify: {} / {}'.format(update.sum(), norm_joints2d.shape[0]))
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif args['run_smplify'] and args['tracking_method'] == 'bbox':
            logging.info('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            logging.info('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args['smooth']:
            min_cutoff = args['smooth_min_cutoff'] # 0.004
            beta = args['smooth_beta'] # 1.5
            logging.info('Running smoothing on person {}, min_cutoff: {}, beta: {}'.format(person_id, min_cutoff, beta))
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'joints2d_img_coord': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    end = time.time()
    fps = num_frames / (end - vibe_time)
    logging.info('VIBE FPS: {:.2f}'.format(fps))

    total_time = time.time() - total_time
    logging.info('Total time spent: {:.2f} seconds (including model loading time).'.format(total_time))
    logging.info('Total FPS (including model loading time): {:.2f}'.format(num_frames / total_time))

    logging.info('Saving output results to \"{}\"'.format(join(output_path, "vibe_output.pkl")))
    joblib.dump(vibe_results, join(output_path, "vibe_output.pkl"))

    if not args['no_render']:

        # for MQTT publish
        payload = {
            'out_vid_full_path': '',
            'out_360_dir_path': []
        }

        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args['wireframe'])

        output_img_folder = f'{image_folder}_output'
        makedirs(output_img_folder, exist_ok=True)

        logging.info('Rendering output video, writing frames to {}'.format(output_img_folder))

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            join(image_folder, x)
            for x in listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if args['sideview']:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                mc = mesh_color[person_id]
                mesh_filename = None

                if args['save_obj']:
                    mesh_folder = join(output_path, 'meshes', f'{person_id:04d}')
                    makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = join(mesh_folder, f'{frame_idx:06d}.obj')

                img_render = renderer.render(img, frame_verts, cam=frame_cam, color=mc, mesh_filename=mesh_filename)

                if args['sideview']:
                    side_img = renderer.render(side_img, frame_verts, cam=frame_cam, color=mc, angle=270, axis=[0,1,0])

                if (frame_idx in args['frame_idx_to_gen_360']):  # frame_idx starts from 0, args['frame_idx_to_gen_360'] starts from 1
                    frame_360_folder = join(output_path, 'frame_{}'.format(frame_idx))
                    # render_a_360_frame(video_file=video_file, vibe_results=vibe_results, frame_idx=frame_idx, output_folder=frame_360_folder, renderer=None, mesh_color=None, frame_results=frame_results) [TODO Crash]
                    for angleValue in range(0, 360, 10):
                        #temp_360_img = np.zeros_like(img)
                        temp_360_img = renderer.render(img, frame_verts, cam=frame_cam, color=mc, angle=angleValue, axis=[0,1,0])
                        makedirs(frame_360_folder, exist_ok=True)
                        cv2.imwrite(join(frame_360_folder, f'{angleValue:06d}.png'), temp_360_img)
                    payload['out_360_dir_path'].append(realpath(frame_360_folder))

            if args['sideview']:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(join(output_img_folder, f'{frame_idx:06d}.png'), img_render)

            # report progress
            if mqttObj != None:
                rendering_video_progress = int(float(frame_idx + 1) / float(len(image_file_names)) * 100)
                progress_payload = {'rendering_video_progress': rendering_video_progress}
                mqttObj.publish(topic, json.dumps(progress_payload))

            if args['display']:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args['display']:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = basename(video_file)
        save_name = '{}_vibe_result.mp4'.format(splitext(vid_name)[0])
        save_name = join(output_path, save_name)
        logging.info('Saving result video to {}'.format(save_name))
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

        payload['out_vid_full_path'] = realpath(save_name)
        if mqttObj != None: mqttObj.publish(topic, json.dumps(payload))

    shutil.rmtree(image_folder)
    logging.info('vibe_3d_pose end')

    return

def render_a_360_frame(video_file, vibe_results, frame_idx, output_folder, renderer=None, mesh_color=None, frame_results=None):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret, img = cap.read()
    orig_height, orig_width = img.shape[0], img.shape[1]
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not frame_results:
        frame_results = prepare_rendering_results(vibe_results, num_frames)

    if not renderer:
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    if not mesh_color:
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    for person_id, person_data in frame_results[frame_idx].items(): # Maybe many people at a frame
        frame_verts = person_data['verts']
        frame_cam = person_data['cam']
        mc = mesh_color[person_id]
        for angleValue in range(0, 360, 10):
            #temp_360_img = np.zeros_like(img)
            temp_360_img = renderer.render(img, frame_verts, cam=frame_cam, color=mc, angle=angleValue, axis=[0,1,0])
            makedirs(output_folder, exist_ok=True)
            cv2.imwrite(join(output_folder, f'{angleValue:06d}.png'), temp_360_img)

class Vibe3DPoseThread(threading.Thread):

    def __init__(self, args, settings):

        threading.Thread.__init__(self)
        self.nodename = args.nodename
        self.broker = settings['mqtt_broker']
        self.waitEvent = threading.Event()
        self.isStopThread = False

        # setup MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker)

        self.vibeArgs = {
            'vid_file': '',  # video file need to render 3D pose. path + file name
            'frame_idx_to_gen_360': [],  # frames in video need to render 3D 360 degrees pose. index starts from 0.
            'output_folder': realpath('./VIBE/vibe_output'),
            'tracking_method': 'bbox',
            'detector': 'yolo',
            'yolo_img_size': int(416),
            'tracker_batch_size': int(3),
            #'tracker_batch_size': int(12),
            'staf_dir': './openposetrack',
            'vibe_batch_size': int(16),
            #'vibe_batch_size': int(450),
            'display': False,
            'run_smplify': False,
            'no_render': False,
            'wireframe': False,
            'sideview': False,
            'save_obj': False,
            'smooth': False,
            'smooth_min_cutoff': float(0.004),
            'smooth_beta': float(0.7)
        }

    def on_connect(self, client, userdata, flag, rc):

        logging.info('Connected with result code: ' + str(rc))
        self.client.subscribe(self.nodename)
        self.client.subscribe('system_control')

    def on_message(self, client, userdata, msg):

        payload = json.loads(msg.payload)

        if 'in_vid_full_path' in payload and 'frame_idx_to_gen_360' in payload:
            self.vibeArgs['vid_file'] = realpath(payload['in_vid_full_path'])
            self.vibeArgs['frame_idx_to_gen_360'] = list(map(int, payload['frame_idx_to_gen_360']))
            self.vibeArgs['output_folder'] = payload['output_folder']
            self.waitEvent.set()

        if 'stop' in payload:
            if payload['stop'] == True:
                self.stop()

    def stop(self):

        logging.info('VIBE node stops')
        self.isStopThread = True
        self.waitEvent.set()

    def run(self):

        logging.info('VIBE node starts')
        # ========= Define VIBE model ========= #
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)

        # ========= Load pretrained weights ========= #
        #pretrained_file = download_ckpt(use_3dpw=False)
        pretrained_file = os.path.join(os.path.expanduser("~"), 'data/vibe_data/vibe_model_wo_3dpw.pth.tar')
        ckpt = torch.load(pretrained_file, map_location=device)
        logging.info('Performance of pretrained model on 3DPW: {}'.format(ckpt["performance"]))
        ckpt = ckpt['gen_state_dict']
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        logging.info('Loaded pretrained weights from \"{}\"'.format(pretrained_file))

        self.client.loop_start()  # MQTT client runs in the background
        sendNodeStateMsg(self.client, self.nodename, "ready")
        while True:
            logging.info('wait for new payload')
            self.waitEvent.wait()  # blocking and wait for valid payload
            self.waitEvent.clear()  # set event flag false for next wait
            if self.isStopThread == True: break

            originalWorkDir = getcwd()
            chdir(dirname(realpath(__file__)))
            vibe_3d_pose(self.vibeArgs, model, self.client, self.nodename)
            chdir(originalWorkDir)

        del model
        sendNodeStateMsg(self.client, self.nodename, "terminated")
        self.client.loop_stop()

def parse_args():

    parser = argparse.ArgumentParser(description='VIBE')
    parser.add_argument('--project', type=str, default='vibe_test.cfg', help='configuration file (default: vibe_test.cfg)')
    parser.add_argument('--nodename', type=str, default='VIBE', help='mqtt node name (default: VIBE)')  # section name in config file
    args = parser.parse_args()
    return args

def main():

    # Parse arguments
    args = parse_args()
    # Load configs
    projectCfg = f"{ROOTDIR}/projects/{args.project}.cfg"
    settings = load_config(projectCfg, args.nodename)

    # start working thread
    thread = Vibe3DPoseThread(args, settings)
    thread.start()
    thread.join()

if __name__ == '__main__':
    main()
