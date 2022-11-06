from cgitb import handler
import logging
import joblib
from math import ceil, sqrt, degrees, acos, dist, pi
import numpy as np
import logging
import numpy as np
import cv2
import json
from numpy.linalg import inv
import configparser
import sys, os
import json
import pandas as pd
import random

class VibePklParser():

    def __init__(self, balltype_dirname, pklFullPath, settings, src_videoname):

        self.pklFullPath = pklFullPath
        self.settings = self.load_camera_config(settings)
        self.dirName = os.path.dirname(self.pklFullPath)
        self.thresholdcfg = configparser.ConfigParser()
        self.thresholdcfg.optionxform = str
        self.thresholdcfg.read(os.path.join(balltype_dirname, "threshold.cfg"))

        info = joblib.load(self.pklFullPath)
        keys = list(info.keys())  # don't know why key changes (sometimes 1, somtimes 4), but there should be only one key (one person)
        self.pklInfo = info[keys[0]]  # dict
        self.jsondict = {}
        self.outputname = os.path.join(os.path.abspath(self.dirName), "analyseResult.json")
        outputfile = open(self.outputname, "w")

        self.tstart_fid, self.tready_fid, self.tstrike_fid, self.tend_fid = self.time_slice() # tstrike is calculated by Model3D.csv
        self.run()
        json.dump(self.jsondict, outputfile, indent = 4, ensure_ascii=False)
        outputfile.close()

        # generate Rwrist video
        output_videoname = os.path.join(self.dirName, "joint.avi")
        self.drawPoint([2,3,4], src_videoname, output_videoname)

        # generate key frame delay video
        key_frame = [self.tstart_fid, self.tready_fid, self.tstrike_fid, self.tend_fid]
        delay_videoname = os.path.join(self.dirName, "delay_video.avi")
        output_picture_path = [os.path.join(self.dirName, "start.jpg"), os.path.join(self.dirName, "ready.jpg"), os.path.join(self.dirName, "strike.jpg"), os.path.join(self.dirName, "end.jpg")]
        self.keyFrameDelayVideo(key_frame, src_videoname, delay_videoname)
        self.keyPostureCropPicture(key_frame, src_videoname, output_picture_path)
        self.storeAll3Dskeleton(os.path.join(self.dirName, "joint3d.csv"))

    def run(self):
        """
        tstart_fid, tready_fid, tstrike_fid, tend_fid = self.time_slice() # tstrike is calculated by Model3D.csv
        print(f'分析影片: {os.path.dirname(pklFullPath)}')
        print(f'時間切割: {tstart_fid}, {tready_fid}, {tstrike_fid}, {tend_fid}')

        print('*****開始姿態分析*****')
        self.jsondict_start = {}
        tstart_score1 = self.tstart_Check1(tstart_fid, float(self.thresholdcfg["start"]["knee_squat_threshold"]), float(self.thresholdcfg["start"]["knee_squat_error"]))
        tstart_score2 = self.tstart_Check2(tstart_fid, float(self.thresholdcfg["start"]["feet_shoulder_same_width_error"]))
        tstart_score3 = self.tstart_Check3(tstart_fid)
        tstart_score4 = self.tstart_Check4(tstart_fid, float(self.thresholdcfg["start"]["lean_angle_threshold"]), float(self.thresholdcfg["start"]["lean_angle_error"]))
        tstart_score = (tstart_score1 + tstart_score2 + tstart_score3 + tstart_score4)*25

        print('*****架拍姿態分析*****')
        self.jsondict_ready = {}
        tready_score1 = self.tready_Check1(tready_fid, float(self.thresholdcfg["ready"]["two_arm_angle_threshold"]), float(self.thresholdcfg["ready"]["two_arm_angle_error"]))
        tready_score2 = self.tready_Check2(tready_fid)
        tready_score3 = self.tready_Check3(tready_fid)
        tready_score = int((tready_score1 + tready_score2 + tready_score3) * 33.3)

        print('*****擊球姿態分析*****')
        self.jsondict_strike = {}
        tstrike_score1 = self.tstrike_Check1(tstrike_fid, float(self.thresholdcfg["strike"]["strike_angle_threshold"]), float(self.thresholdcfg["strike"]["strike_angle_error"]))
        tstrike_score2 = self.tstrike_Check2(tstrike_fid, float(self.thresholdcfg["strike"]["elbow_angle_threshold"]), float(self.thresholdcfg["strike"]["elbow_angle_error"]))
        tstrike_score3 = self.tstrike_Check3(float(self.thresholdcfg["strike"]["speed_threshold"]))
        tstrike_score = int((tstrike_score1 + tstrike_score2 + tstrike_score3) * 33.3)

        print('*****結束姿態分析*****')
        self.jsondict_end = {}
        tend_score1 = self.tend_Check1(tend_fid, float(self.thresholdcfg["end"]["Rwrist_front_LHip_threshold"]), float(self.thresholdcfg["end"]["Rwrist_front_LHip_error"]))
        tend_score2 = self.tend_Check2(tend_fid)
        tend_score = (tend_score1 + tend_score2) * 50

        print('\n\n')
        self.jsondict = {}
        self.jsondict["tstart"] = self.jsondict_start
        self.jsondict["tstart"]["frame"] = tstart_fid
        self.jsondict["tstart"]["score"] = {
            "text": "總評:",
            "value": tstart_score
        }

        self.jsondict["tready"] = self.jsondict_ready
        self.jsondict["tready"]["frame"] = tready_fid
        self.jsondict["tready"]["score"] = {
            "text": "總評:",
            "value": tready_score
        }

        self.jsondict["tstrike"] = self.jsondict_strike
        self.jsondict["tstrike"]["frame"] = tstrike_fid
        self.jsondict["tstrike"]["score"] =  {
            "text": "總評:",
            "value": tstrike_score
        }

        self.jsondict["tend"] = self.jsondict_end
        self.jsondict["tend"]["frame"] = tend_fid
        self.jsondict["tend"]["score"] =  {
            "text": "總評:",
            "value": tend_score
        }
        """
        raise NotImplementedError()

    def time_slice(self):
        """
        # use Model3D.csv to get T_strike
        model3D_path = os.path.join(os.path.dirname(self.dirName), "Model3D.csv")
        print(f"model3D_path: {model3D_path}")
        camerareaderL_path = os.path.join(os.path.dirname(self.dirName), "CameraReaderL.csv")
        # Get fps
        cap = cv2.VideoCapture(os.path.join(os.path.dirname(self.dirName), "CameraReaderL.avi"))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        df_camerareaderL = pd.read_csv(camerareaderL_path)
        video_start_frame = df_camerareaderL['Frame'][0]
        df_model3D = pd.read_csv(model3D_path)
        for i in range(df_model3D.shape[0]):
            if df_model3D['Event'][i] == 1:
                video_hit_frame = df_model3D['Frame'][i]
                if i+1 < df_model3D.shape[0]:
                    self.speed = ((float(df_model3D['X'][i+1]) - float(df_model3D['X'][i]))**2 + (float(df_model3D['Y'][i+1]) - float(df_model3D['Y'][i]))**2 + (float(df_model3D['Z'][i+1]) - float(df_model3D['Z'][i]))**2 )**0.5
                else:
                    self.speed = random.uniform(80, 130)
                    logging.error('t_strike speed is random')
                break
        tstrike_fid = int(video_hit_frame - video_start_frame)

        ### find out T_ready
        joints3d_length = self.pklInfo['joints3d'].shape[0]
        rwrist_court_ymax = float('-inf')
        for frameidx in range(int(joints3d_length/2)):
            #logging.debug('FID: {} cor_3d_rwrist: {}'.format(frameidx, self.get3DSKP_court(frameidx, 4).tolist()))
            listidx = self.getListIdxByFrameIdx(frameidx)
            if self.get3DSKP_court(listidx, 3).tolist()[1] > rwrist_court_ymax:
                rwrist_court_ymax = self.get3DSKP_court(listidx, 3).tolist()[1]
                t_ready = frameidx
        # logging.debug(f't_ready : {t_ready}')

        ### use 2d skeleton to find out T_start
        joints2d_length = self.pklInfo['joints2d_img_coord'].shape[0]
        list_skeleton2d_diff_sum = []
        for frameidx in range(int(joints2d_length/2)):
            skeleton2d_diff_sum = 0
            listidx = self.getListIdxByFrameIdx(frameidx)
            for listidx_iter in range(listidx,listidx+1):
                for kp_idx in [4]:
                    skeleton2d_diff_sum += np.linalg.norm(self.pklInfo['joints2d_img_coord'][listidx_iter + 1][kp_idx] - self.pklInfo['joints2d_img_coord'][listidx_iter][kp_idx])
            logging.debug(f'fid: {frameidx} skeleton2d_diff_sum: {skeleton2d_diff_sum}')
            list_skeleton2d_diff_sum.append(skeleton2d_diff_sum)
        t_start_threshold = np.percentile(np.array(list_skeleton2d_diff_sum), 30)
        for i in range(len(list_skeleton2d_diff_sum)):
            if list_skeleton2d_diff_sum[i] < t_start_threshold:
                list_skeleton2d_diff_sum[i] = 0 # no move
            else:
                list_skeleton2d_diff_sum[i] = 1 # move
        # print(list_skeleton2d_diff_sum)
        have_t_start = False
        t_start = 0
        for i in range(len(list_skeleton2d_diff_sum)-6):
            if (list_skeleton2d_diff_sum[i]==0) & (list_skeleton2d_diff_sum[i+1]==0) & (list_skeleton2d_diff_sum[i+2]==0) & (list_skeleton2d_diff_sum[i+3]==0) & (list_skeleton2d_diff_sum[i+4]==0) & (list_skeleton2d_diff_sum[i+5]==1) & (list_skeleton2d_diff_sum[i+6]==1):
                have_t_start = True
                t_start = i+4
                # print('have_t_start')

        for frameidx in range(int(joints3d_length/2)):
            skeleton_diff_sum = 0
            listidx = self.getListIdxByFrameIdx(frameidx)
            for listidx_iter in range(listidx,listidx+1): # total S frame
                for kp_idx in [4]: # total K skeleton keypoint
                    skeleton_diff_sum += np.linalg.norm(self.get3DSKP_court(listidx_iter + 1, kp_idx) - self.get3DSKP_court(listidx_iter, kp_idx))
            #logging.debug(f'fid: {frameidx} skeleton_diff_sum: {skeleton_diff_sum}')

        ### find out T_end
        rwrist_court_zmin = float('inf')
        t_end = 0
        for frameidx in range(tstrike_fid, int(joints3d_length) - 1, 1):
            listidx = self.getListIdxByFrameIdx(frameidx)
            if self.get3DSKP_court(listidx, 4).tolist()[2] < rwrist_court_zmin:
                rwrist_court_zmin = self.get3DSKP_court(listidx, 4).tolist()[2]
                t_end = frameidx
        #logging.debug(f't_end : {t_end}')
        if t_end == 0:
            print("t_end error")
        self.t_start = t_start
        self.t_ready = t_ready
        self.t_end = t_end
        return (t_start, t_ready, tstrike_fid, t_end)
        """
        raise NotImplementedError()

    def getJointNames(self):

        return ['OP Nose',       # 0
                'OP Neck',       # 1
                'OP RShoulder',  # 2
                'OP RElbow',     # 3
                'OP RWrist',     # 4
                'OP LShoulder',  # 5
                'OP LElbow',     # 6
                'OP LWrist',     # 7
                'OP MidHip',     # 8
                'OP RHip',       # 9
                'OP RKnee',      # 10
                'OP RAnkle',     # 11
                'OP LHip',       # 12
                'OP LKnee',      # 13
                'OP LAnkle',     # 14
                'OP REye',       # 15
                'OP LEye',       # 16
                'OP REar',       # 17
                'OP LEar',       # 18
                'OP LBigToe',    # 19
                'OP LSmallToe',  # 20
                'OP LHeel',      # 21
                'OP RBigToe',    # 22
                'OP RSmallToe',  # 23
                'OP RHeel',      # 24
                'rankle',        # 25
                'rknee',         # 26
                'rhip',          # 27
                'lhip',          # 28
                'lknee',         # 29
                'lankle',        # 30
                'rwrist',        # 31
                'relbow',        # 32
                'rshoulder',     # 33
                'lshoulder',     # 34
                'lelbow',        # 35
                'lwrist',        # 36
                'neck',          # 37
                'headtop',       # 38
                'hip',           # 39 'Pelvis (MPII)', # 39
                'thorax',        # 40 'Thorax (MPII)', # 40
                'Spine (H36M)',  # 41
                'Jaw (H36M)',    # 42
                'Head (H36M)',   # 43
                'nose',          # 44
                'leye',          # 45 'Left Eye', # 45
                'reye',          # 46 'Right Eye', # 46
                'lear',          # 47 'Left Ear', # 47
                'rear',          # 48 'Right Ear', # 48
            ]

##### Library
    def getListIdxByFrameIdx(self, Idx):
        for listIdx, frameIdx in enumerate(self.pklInfo['frame_ids']):
            if Idx == frameIdx:
                return listIdx
        return -1

    def getSkeleton2D(self, keypoint_idx):
        joints2d_length = self.pklInfo['joints2d_img_coord'].shape[0]
        cor_2d = []
        for frameidx in range(joints2d_length):
            cor_2d.append(self.pklInfo['joints2d_img_coord'][frameidx][keypoint_idx])
        return np.array(cor_2d)

    def get3DSKP_court(self, fid, keypoint_idx):
        ''' This function output 3D coordinate of pose keypoint in court coordinate space

        Args:
            frameidx (int): The frame index of (2s) video clip.
            keypoint_idx (int): The keypoint index in self.getJointNames.

        Returns:
            <class 'np.ndarray'> shape: (3)
            3D coordinate of pose keypoint in court coordinate space
        '''

        listidx = self.getListIdxByFrameIdx(fid)
        if (listidx == -1):
            raise ValueError("listidx must be a postive integer")
        objPoints = []
        imgPoints = []
        for i in range(49):
            objPoints.append(self.pklInfo['joints3d'][listidx][i])
            imgPoints.append(self.pklInfo['joints2d_img_coord'][listidx][i])
        objPoints = np.array(objPoints)
        imgPoints = np.array(imgPoints)
        cameraMatrix = np.array(json.loads(self.settings['Other']['ks']),np.float32)
        distCoeffs = np.array(json.loads(self.settings['Other']['dist']),np.float32)
        retval, rvec, tvec  = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, flags = cv2.SOLVEPNP_EPNP)
        pose_R = np.array(cv2.Rodrigues(rvec)[0],np.float32)
        pose_T = np.array(tvec,np.float32).reshape(3,1)

        ### load court coordinate R and T (extrinsic matrix)
        extrinsic_mat = np.array(json.loads(self.settings['Other']['extrinsic_mat']),np.float32)
        court_R = extrinsic_mat[:,0:3]
        court_T = extrinsic_mat[:,3].reshape(3,1)

        nose_3d = np.array(self.pklInfo['joints3d'][listidx][keypoint_idx]).reshape(3,1)
        nose_3d_court_cor = inv(court_R) @ (pose_R @ nose_3d + pose_T - court_T)
        return nose_3d_court_cor.reshape(3)

    def getAngle(self, point_a, point_b, point_c, point_d):
        vector1 = point_b - point_a
        vector2 = point_d - point_c
        return degrees(np.arccos(np.dot(vector1, vector2) / (sqrt(np.sum(np.power(vector1,2))) * sqrt(np.sum(np.power(vector2,2))))))

    def get3DSKP_pose(self, fid, keypoint_idx): # get skeleton 3D point in pose coordinate
        listidx = self.getListIdxByFrameIdx(fid)
        skeleton3D = self.pklInfo['joints3d'][listidx][keypoint_idx]
        return skeleton3D

    def compare(self, point_a, point_b, point_c, point_d): # location compare : point_a is front/inside/behind of the plane BCD
        vector1 = point_c - point_b
        vector2 = point_d - point_c
        normal_vec = np.cross(vector1.reshape(3), vector2.reshape(3))
        if np.dot(normal_vec, point_b - point_a) < 0:
            return 1 # front
        elif np.dot(normal_vec, point_b - point_a) == 0:
            return 0 # inside
        else:
            return -1 # behind

    def getGravityPoint(self, fid, ismale):
        head = (self.get3DSKP_court(fid, 17) + self.get3DSKP_court(fid, 18))/2
        trunk = (self.get3DSKP_court(fid, 2) + self.get3DSKP_court(fid, 5) + self.get3DSKP_court(fid, 9) + self.get3DSKP_court(fid, 12))/4
        arm = (self.get3DSKP_court(fid, 2) + self.get3DSKP_court(fid, 3) + self.get3DSKP_court(fid, 5) + self.get3DSKP_court(fid, 6))/4
        forearm = (self.get3DSKP_court(fid, 3) + self.get3DSKP_court(fid, 4) + self.get3DSKP_court(fid, 6) + self.get3DSKP_court(fid, 7))/4
        hand = (self.get3DSKP_court(fid, 4) + self.get3DSKP_court(fid, 7))/2
        thigh = (self.get3DSKP_court(fid, 9) + self.get3DSKP_court(fid, 10) + self.get3DSKP_court(fid, 12) + self.get3DSKP_court(fid, 13))/4
        lower_leg = (self.get3DSKP_court(fid, 10) + self.get3DSKP_court(fid, 11) + self.get3DSKP_court(fid, 13) + self.get3DSKP_court(fid, 14))/4
        foot = (self.get3DSKP_court(fid, 11) + self.get3DSKP_court(fid, 22) + self.get3DSKP_court(fid, 14) + self.get3DSKP_court(fid, 19))/4
        if ismale == 1:
            gravity_point = (head*8.26 + trunk*46.84 + arm*3.25 + forearm*1.87 + hand*0.65 + thigh*10.5 + lower_leg*4.75 + foot*1.43) / 77.55
        elif ismale == 0:
            gravity_point = (head*8.2 + trunk*45 + arm*2.9 + forearm*1.57 + hand*0.5 + thigh*11.75 + lower_leg*5.35 + foot*1.33) / 76.6
        else:
            gravity_point = (head*8.23 + trunk*45.92 + arm*3.07 + forearm*1.72 + hand*0.57 + thigh*11.12 + lower_leg*5.05 + foot*1.38) / 77.06
        return gravity_point



##### Additional info
    def drawPoint(self, skeletonID_list, input_video_path, output_video_path):
        video_sk_cor2d = []
        for i in skeletonID_list:
            video_sk_cor2d.append(np.floor(self.getSkeleton2D(i)))
        cap = cv2.VideoCapture(input_video_path)
        output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))
        index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            # if frame is read correctly ret is True
            for i in range(len(skeletonID_list)):
                frame = cv2.circle(frame, (int(video_sk_cor2d[i][index][0]), int(video_sk_cor2d[i][index][1])), radius=5, color=(0, 0, 255), thickness=-1)
            writer.write(frame)
            index += 1
        cap.release()

    def plotLine(self, subfig, frameidx, kp1, kp2, origin):
        if origin == 1:
            subfig.plot([self.pklInfo['joints3d'][frameidx][kp1][0], self.pklInfo['joints3d'][frameidx][kp2][0]], [self.pklInfo['joints3d'][frameidx][kp1][1], self.pklInfo['joints3d'][frameidx][kp2][1]], [self.pklInfo['joints3d'][frameidx][kp1][2], self.pklInfo['joints3d'][frameidx][kp2][2]], marker='o')
        else:
            subfig.plot([self.get3DSKP_court(0, kp1).tolist()[0],self.get3DSKP_court(0, kp2).tolist()[0]], [self.get3DSKP_court(0, kp1).tolist()[1], self.get3DSKP_court(0, kp2).tolist()[1]], [self.get3DSKP_court(0, kp1).tolist()[2], self.get3DSKP_court(0, kp2).tolist()[2]], marker='o')

    def keyFrameDelayVideo(self, keyframe, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))
        index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            # if frame is read correctly ret is True
            writer.write(frame)
            if index in keyframe:
                for i in range(int(fps*1.2)):
                    writer.write(frame)
            index += 1
        cap.release()

    def keyPostureCropPicture(self, keyframe, input_video_path, output_picture_path):
        cap = cv2.VideoCapture(input_video_path)
        output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        index = 0
        keyframe_list = []
        logging.warning(f'keyframe: {keyframe}')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            if index in keyframe:
                keyframe_list.append(frame)
            index += 1
        cap.release()

        for n, frameidx in enumerate(keyframe):
            right = 0
            top = output_height-1
            left = output_width-1
            buttom = 0
            listidx = self.getListIdxByFrameIdx(frameidx)
            for keypoint_idx in range(25):
                x, y = self.pklInfo['joints2d_img_coord'][listidx][keypoint_idx]
                if x < left:
                    left = x
                elif x > right:
                    right = x
                if y < top:
                    top = y
                elif y > buttom:
                    buttom = y
            if top - 100 > 0:
                top = top - 100
            if buttom + 100 < output_height:
                buttom = buttom + 100
            if left - 100 > 0:
                left = left - 100
            if right + 100 < output_width:
                right = right + 100
            keyframe_list[n] = keyframe_list[n][int(top):int(buttom), int(left):int(right)]
            cv2.imwrite(output_picture_path[n], keyframe_list[n])

    def storeAll3Dskeleton(self, outputpath):
        pointList = []
        self.totalframe = self.pklInfo['joints3d'].shape[0]
        for frameidx in range(int(self.totalframe)):
            for i in range(25):
                pointList.append([self.get3DSKP_court(frameidx, i).tolist()[0], self.get3DSKP_court(frameidx, i).tolist()[1], self.get3DSKP_court(0, i).tolist()[2]])
            pointList.append(self.getGravityPoint(frameidx, 1))
        df = pd.DataFrame(pointList, columns = ["x", "y", "z"])
        df.to_csv(outputpath)

    def load_camera_config(self, cfg):
        try:
            config = configparser.ConfigParser()
            config.optionxform = str
            config.read(cfg)
        except IOError as e:
            logging.error(e)
            sys.exit()
        return config