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
sys.path.append(f"{os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'lib')}")
from VibePklParser import VibePklParser

DIRNAME = os.path.dirname(os.path.abspath(__file__))

class Backhand_Cut_VibePklParser(VibePklParser):
    def __init__(self, balltype_dirname, pklFullPath, settings, src_videoname):
        super().__init__(balltype_dirname, pklFullPath, settings, src_videoname)

    def run(self):
        tstart_fid, tready_fid, tstrike_fid, tend_fid = self.tstart_fid, self.tready_fid, self.tstrike_fid, self.tend_fid # tstrike is calculated by Model3D.csv
        logging.debug(f'分析影片: {os.path.dirname(self.pklFullPath)}')
        logging.debug(f'時間切割: {tstart_fid}, {tready_fid}, {tstrike_fid}')

        logging.debug('*****開始姿態分析*****')
        self.jsondict_start = {}
        tstart_score1 = self.tstart_Knee_Sqaut(tstart_fid, float(self.thresholdcfg["start"]["knee_squat_threshold"]), float(self.thresholdcfg["start"]["knee_squat_error"]))
        tstart_score2 = self.tstart_FS_Same_Width(tstart_fid, float(self.thresholdcfg["start"]["feet_shoulder_same_width_error"]))
        tstart_score3 = self.tstart_Two_Hand_Front_Body(tstart_fid)
        tstart_score4 = self.tstart_Lean_Angle(tstart_fid, float(self.thresholdcfg["start"]["lean_angle_threshold"]), float(self.thresholdcfg["start"]["lean_angle_error"]))
        tstart_score = (tstart_score1 + tstart_score2 + tstart_score3 + tstart_score4)*25

        logging.debug('*****架拍姿態分析*****')
        self.jsondict_ready = {}
        tready_score1 = self.tready_Elbow_Higher_Than_Chest(tready_fid)
        tready_score2 = self.tready_Back_To_Court(tready_fid)
        tready_score3 = self.tready_Weight_On_Right_Foot(tready_fid)
        tready_score = int((tready_score1 + tready_score2 + tready_score3) * 33.3)

        logging.debug('*****擊球姿態分析*****')
        self.jsondict_strike = {}
        tstrike_score1 = self.tstrike_Strike_Angle(tstrike_fid, float(self.thresholdcfg["strike"]["strike_angle_threshold"]), float(self.thresholdcfg["strike"]["strike_angle_error"]))
        tstrike_score2 = self.tstrike_Elbow_Angle(tstrike_fid, float(self.thresholdcfg["strike"]["elbow_angle_threshold"]), float(self.thresholdcfg["strike"]["elbow_angle_error"]))
        tstrike_score3 = self.tstrike_Speed(float(self.thresholdcfg["strike"]["speed_threshold"]))
        tstrike_score = int((tstrike_score1 + tstrike_score2 + tstrike_score3) * 33.3)

        logging.debug('*****結束姿態分析*****')
        self.jsondict_end = {}

        logging.debug('\n\n')
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
            "value": "無分析資料"
        }

    def time_slice(self):

        # use Model3D.csv to get T_strike
        model3D_path = os.path.join(os.path.dirname(self.dirName), "Model3D.csv")
        logging.debug(f"model3D_path: {model3D_path}")
        camerareaderL_path = os.path.join(os.path.dirname(self.dirName), "CameraReaderL.csv")
        # Get fps
        cap = cv2.VideoCapture(os.path.join(os.path.dirname(self.dirName), "CameraReaderL.avi"))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        df_camerareaderL = pd.read_csv(camerareaderL_path)
        video_start_frame = df_camerareaderL['Frame'][0]
        df_model3D = pd.read_csv(model3D_path)
        ### find out T_strike
        ymax = -1000.0
        video_hit_frame = 0
        for i in range(df_model3D.shape[0]-1):
            if float(df_model3D['Y'][i]) > ymax:
                ymax = float(df_model3D['Y'][i])
                video_hit_frame = df_model3D['Frame'][i]

        logging.warning(f'video_hit_frame: {video_hit_frame}')
        if video_hit_frame+1 < df_model3D.shape[0]:
            self.speed = ((float(df_model3D['X'][video_hit_frame+1]) - float(df_model3D['X'][video_hit_frame]))**2 + (float(df_model3D['Y'][video_hit_frame+1]) - float(df_model3D['Y'][video_hit_frame]))**2 + (float(df_model3D['Z'][video_hit_frame+1]) - float(df_model3D['Z'][video_hit_frame]))**2 )**0.5 / (float(df_model3D['Timestamp'][video_hit_frame+1]) - float(df_model3D['Timestamp'][video_hit_frame])) * 3.6
        else:
            self.speed = random.uniform(80, 130)
            logging.error('t_strike speed is random')
        tstrike_fid = int(video_hit_frame - video_start_frame)
        logging.debug(f'tstrike_fid: {tstrike_fid}')

        ### find out T_ready
        joints3d_length = self.pklInfo['joints3d'].shape[0]
        t_ready = 0
        rwrist_court_ymin = float('-inf')
        for frameidx in range(tstrike_fid):
            #logging.debug('FID: {} cor_3d_rwrist: {}'.format(frameidx, self.get3DSKP_court(frameidx, 4).tolist()))
            listidx = self.getListIdxByFrameIdx(frameidx)
            RWrist = self.get3DSKP_court(listidx, 4)
            RElbow = self.get3DSKP_court(listidx, 3)
            Rshoulder = self.get3DSKP_court(listidx, 2)
            Lshoulder = self.get3DSKP_court(listidx, 5)
            Neck = self.get3DSKP_court(listidx, 1)
            if ((np.linalg.norm(RWrist - Rshoulder) > np.linalg.norm(RWrist - Lshoulder))):
                logging.debug(f'frame {frameidx} 手腕在身體左側')
            if (RElbow[2] > Neck[2]):
                logging.debug(f'frame {frameidx} 手腕高於脖子')
            if (RElbow[2] > Neck[2]) & (RWrist[1] > rwrist_court_ymin): # 反拍的架拍手腕需靠左邊
                rwrist_court_ymin = RWrist[1]
                t_ready = frameidx
        if (t_ready==0):
            logging.error('No T_ready')
            t_ready = 30
        logging.debug(f't_ready : {t_ready}')

        ### use 2d skeleton to find out T_start
        joints2d_length = self.pklInfo['joints2d_img_coord'].shape[0]
        list_skeleton2d_diff_sum = []
        for frameidx in range(t_ready):
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
        # logging.debug(list_skeleton2d_diff_sum)
        have_t_start = False
        t_start = 0
        for i in range(len(list_skeleton2d_diff_sum)-6):
            if (list_skeleton2d_diff_sum[i]==0) & (list_skeleton2d_diff_sum[i+1]==0) & (list_skeleton2d_diff_sum[i+2]==0) & (list_skeleton2d_diff_sum[i+3]==0) & (list_skeleton2d_diff_sum[i+4]==0) & (list_skeleton2d_diff_sum[i+5]==1) & (list_skeleton2d_diff_sum[i+6]==1):
                have_t_start = True
                t_start = i+4
                # logging.debug('have_t_start')
        if t_start<6:
            t_start = 10

        ### find out T_end
        rwrist_court_zmin = float('inf')
        t_end = 0
        for frameidx in range(tstrike_fid, self.video_length - 1, 1):
            listidx = self.getListIdxByFrameIdx(frameidx)
            if self.get3DSKP_court(listidx, 4).tolist()[2] < rwrist_court_zmin:
                rwrist_court_zmin = self.get3DSKP_court(listidx, 4).tolist()[2]
                t_end = frameidx

        self.t_start = t_start
        self.t_ready = t_ready
        self.t_end = t_end
        return (t_start, t_ready, tstrike_fid, t_end)

    ##### Analyze
    ### t_start
    def tstart_Knee_Sqaut(self, tstart_fid, threshold, error): # 膝蓋微蹲 knee_sqaut
        RHip = self.get3DSKP_pose(tstart_fid, 9)
        RKnee = self.get3DSKP_pose(tstart_fid, 10)
        RAnkle = self.get3DSKP_pose(tstart_fid, 11)
        logging.debug(f'右腳膝蓋夾角: {self.getAngle(RKnee, RHip, RKnee, RAnkle)}')
        LHip = self.get3DSKP_pose(tstart_fid, 12)
        LKnee = self.get3DSKP_pose(tstart_fid, 13)
        LAnkle = self.get3DSKP_pose(tstart_fid, 14)
        logging.debug(f'左腳膝蓋夾角: {self.getAngle(LKnee, LHip, LKnee, LAnkle)}')
        knee_sqaut_dict = {}
        knee_sqaut_dict["text"] = "膝蓋微彎:"
        knee_sqaut_dict["value"] = {
                "RKnee_angle": {
                    "text": "右膝蓋彎曲:",
                    "value": self.getAngle(RKnee, RHip, RKnee, RAnkle)
                },
                "LKnee_angle": {
                    "text": "左膝蓋彎曲:",
                    "value": self.getAngle(LKnee, LHip, LKnee, LAnkle)
                }
        }

        if ((abs(self.getAngle(RKnee, RHip, RKnee, RAnkle) - threshold) < error) & (abs(self.getAngle(LKnee, LHip, LKnee, LAnkle) - threshold) < error)):
            knee_sqaut_dict["pass"] = True
            self.jsondict_start["knee_sqaut"] = knee_sqaut_dict
            return 1
        else:
            knee_sqaut_dict["pass"] = False
            self.jsondict_start["knee_sqaut"] = knee_sqaut_dict
            return 0

    def tstart_FS_Same_Width(self, tstart_fid, error): # 雙腳與肩同寬
        shoulder_width = dist(self.get3DSKP_court(tstart_fid, 2), self.get3DSKP_court(tstart_fid, 5))
        feet_width = dist(self.get3DSKP_court(tstart_fid, 21), self.get3DSKP_court(tstart_fid, 24))
        logging.debug(f'肩膀寬度: {shoulder_width}     雙腳寬度: {feet_width}')
        feet_shoulder_same_width_dict = {}
        feet_shoulder_same_width_dict["text"] = "雙腳與肩同寬:"
        feet_shoulder_same_width_dict["value"] = {
                "feet_width": {
                    "text": "雙腳寬:",
                    "value": feet_width
                },
                "shoulder_width": {
                    "text": "肩膀寬:",
                    "value": shoulder_width
                }
            }
        if abs(shoulder_width - feet_width) < error:
            feet_shoulder_same_width_dict["pass"] = True
            self.jsondict_start["feet_shoulder_same_width"] = feet_shoulder_same_width_dict
            return 1
        else:
            feet_shoulder_same_width_dict["pass"] = False
            self.jsondict_start["feet_shoulder_same_width"] = feet_shoulder_same_width_dict
            return 0

    def tstart_Two_Hand_Front_Body(self, tstart_fid): # 雙手自然放於身前 [TODO]
        LShoulder = self.get3DSKP_court(tstart_fid, 5)
        RShoulder = self.get3DSKP_court(tstart_fid, 2)
        RHip = self.get3DSKP_court(tstart_fid, 9)
        RWrist = self.get3DSKP_court(tstart_fid, 4)
        LWrist = self.get3DSKP_court(tstart_fid, 7)
        two_hand_front_body_dict = {}
        two_hand_front_body_dict["text"] = "雙手於胸前方:"
        if ((self.compare(RWrist, LShoulder, RShoulder, RHip)==1) & (self.compare(LWrist, LShoulder, RShoulder, RHip)==1)):
            two_hand_front_body_dict["pass"] = True
            self.jsondict_start["two_hand_front_body"] = two_hand_front_body_dict
            return 1
        else:
            two_hand_front_body_dict["pass"] = False
            self.jsondict_start["two_hand_front_body"] = two_hand_front_body_dict
            return 0

    def tstart_Lean_Angle(self, tstart_fid, threshold, error): # 身體前傾
        Neck = self.get3DSKP_court(tstart_fid, 1)
        MidHip = self.get3DSKP_court(tstart_fid, 8)
        lean_angle = self.getAngle(MidHip, Neck, np.array([0,0,0]), np.array([0,0,1]))
        lean_angle_dict = {}
        lean_angle_dict["text"] = "身體微微前傾:"
        lean_angle_dict["value"] = lean_angle
        if (abs(self.getAngle(MidHip, Neck, np.array([0,0,0]), np.array([0,0,1])) - threshold) < error):
            lean_angle_dict["pass"] = True
            self.jsondict_start["lean_angle"] = lean_angle_dict
            return 1
        else:
            lean_angle_dict["pass"] = False
            self.jsondict_start["lean_angle"] = lean_angle_dict
            return 0

    ### t_ready
    def tready_Elbow_Higher_Than_Chest(self, tready_fid): # 手肘高於胸
        RElbow = self.get3DSKP_court(tready_fid, 3)
        Chest = (self.get3DSKP_court(tready_fid, 1) + self.get3DSKP_court(tready_fid, 8))/2
        if (RElbow[2] > Chest[2]):
            self.jsondict_ready["elbow_higher_than_chest"] = {
                "text": "慣用手肘高於胸:",
                "pass": True
            }
            return 1
        else:
            self.jsondict_ready["elbow_higher_than_chest"] = {
                "text": "慣用手肘高於胸:",
                "pass": False
            }
            return 0

    def tready_Back_To_Court(self, tready_fid): # 身體背對球網
        RShoulder = self.get3DSKP_court(tready_fid, 2)
        LShoulder = self.get3DSKP_court(tready_fid, 5)
        Hip = self.get3DSKP_court(tready_fid, 8)
        face_vec = np.cross(RShoulder-LShoulder, Hip-LShoulder)
        if(np.dot(face_vec, np.array([0,0,1])) < 0):
            self.jsondict_ready["back_to_court"] = {
                "text": "身體背對球網:",
                "pass": True
            }
            return 1
        else:
            self.jsondict_ready["back_to_court"] = {
                "text": "身體背對球網:",
                "pass": False
            }
            return 0

    def tready_Weight_On_Right_Foot(self, tready_fid): # 重心在右腳
        gravitypoint = self.getGravityPoint(tready_fid, 2)
        gravitypoint[2] = 0
        LHeel = self.get3DSKP_court(tready_fid, 21)
        RHeel = self.get3DSKP_court(tready_fid, 24)
        if np.linalg.norm(gravitypoint - RHeel)*3 < np.linalg.norm(gravitypoint - LHeel):
            self.jsondict_ready["weight_on_right_foot"] = {
                "text": "重心在右腳:",
                "pass": True
            }
            return 1
        else:
            self.jsondict_ready["weight_on_right_foot"] = {
                "text": "重心在右腳:",
                "pass": False
            }
            return 0
    ### t_strike
    def tstrike_Strike_Angle(self, tstrike_fid, threshold, error): # 擊球角
        RShoulder = self.get3DSKP_court(tstrike_fid, 2)
        RWrist = self.get3DSKP_court(tstrike_fid, 4)
        hitballangle = self.getAngle(RShoulder, RWrist, np.array([0,0,0]), np.array([0,0,1]))
        if abs(hitballangle - threshold) < error:
            self.jsondict_strike["strike_angle"] = {
                "text": "擊球角:",
                "pass": True,
                "value": hitballangle
            }
            return 1
        else:
            self.jsondict_strike["strike_angle"] = {
                "text": "擊球角:",
                "pass": False,
                "value": hitballangle
            }
            return 0

    def tstrike_Elbow_Angle(self, tstrike_fid, threshold, error): # 手肘角
        RShoulder = self.get3DSKP_pose(tstrike_fid, 2)
        RElbow = self.get3DSKP_pose(tstrike_fid, 3)
        RWrist = self.get3DSKP_pose(tstrike_fid, 4)
        elbowangle =  self.getAngle(RElbow, RShoulder, RElbow, RWrist)
        if abs(elbowangle - threshold) < error:
            self.jsondict_strike["elbow_angle"] = {
                "text": "手肘張開:",
                "pass": True,
                "value": elbowangle
            }
            return 1
        else:
            self.jsondict_strike["elbow_angle"] = {
                "text": "手肘張開:",
                "pass": False,
                "value": elbowangle
            }
            return 0

    def tstrike_Speed(self, threshold): # 球速
        if self.speed > threshold:
            self.jsondict_strike["speed"] = {
                    "text": "最高球速:",
                    "pass": True,
                    "value": self.speed
            }
            return 1
        else:
            self.jsondict_strike["speed"] = {
                    "text": "最高球速:",
                    "pass": False,
                    "value": self.speed
            }
            return 0

def main():
    pklparser = Backhand_Cut_VibePklParser(DIRNAME, sys.argv[1], sys.argv[2], sys.argv[3]) # pkl, camera cfg, src videoname
    slice_pair = pklparser.time_slice()
    return slice_pair

if __name__ == '__main__':
    main()
