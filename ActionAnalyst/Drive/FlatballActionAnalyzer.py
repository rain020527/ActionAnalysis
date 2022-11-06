import argparse
import json
import math
import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from numpy.linalg import norm


class FlatballActionAnalyzer:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        videoPath="CameraReaderL.avi",
        trackNetCSV="TrackNetL.csv",
        output_folder = '.',
        debugging = False
    ):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.debugging = debugging
        self.videoPath = videoPath
        self.df = pd.read_csv(trackNetCSV)
        self.cap = cv2.VideoCapture(self.videoPath)
        self.output_folder = os.path.join(output_folder, os.path.basename(videoPath).replace('.avi', '_Analyze'))
        os.makedirs(self.output_folder, exist_ok=True)
        self.elbowAngleList = []

        ######
        self.NOSE = 0,
        self.LEFT_EYE_INNER = 1
        self.LEFT_EYE = 2
        self.LEFT_EYE_OUTER = 3
        self.RIGHT_EYE_INNER = 4
        self.RIGHT_EYE = 5
        self.RIGHT_EYE_OUTER = 6
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        self.MOUTH_LEFT = 9
        self.MOUTH_RIGHT = 10
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_PINKY = 17
        self.RIGHT_PINKY = 18
        self.LEFT_INDEX = 19
        self.RIGHT_INDEX = 20
        self.LEFT_THUMB = 21
        self.RIGHT_THUMB = 22
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_HEEL = 29
        self.RIGHT_HEEL = 30
        self.LEFT_FOOT_INDEX = 31
        self.RIGHT_FOOT_INDEX = 32
        ######
        #print("INIT")

        self.result = dict()
        self.pixel2meter = np.array([1080,1440])
        self.result = dict()
        self.getJointPos()
        self.analyse()

        return

    def getJointPos(self):
        id = 0
        while True:
            ret, img = self.cap.read()
            if ret == False:
                break
            img = self.connectLandmarks(img)
            self.elbowAngleList.append(self.getElbowAngle(img))
            if self.debugging == True:
                cv2.imshow("Image", img)
                cv2.waitKey(1)
        hitPoint = self.getHitPointFrame(self.df)
        backSwingPoint = self.getbackSwingPointFrame(hitPoint)
        preparePoint = self.getPreparePointFrame(self.cap, backSwingPoint)
        self.writeResults(self.cap, hitPoint,
                          backSwingPoint, preparePoint)
        self.preparePoint = preparePoint
        self.backSwingPoint = backSwingPoint
        self.hitPoint = hitPoint
        return

    def connectLandmarks(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )
        return img

    def getLandmarkPoint(self, img, drawCircle = True):
        dict = {}
        h, w, c = img.shape
        self.w = w # for getHitpoint()
        # img not None
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imgRGB not None
        self.results = self.pose.process(imgRGB)
        # self.results is None
        #print(f"results {self.results.pose_landmarks}")
        if self.results.pose_landmarks != None:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                dict[id] = (int(lm.x*w), int(lm.y*h))
                if drawCircle == True:
                    cv2.circle(img, dict[id], 5, (255, 0, 0), cv2.FILLED)
        return dict

    def markElbowAngle(self, img):
        cv2.line(img, self.dict[14], self.dict[12], (0, 0, 255), 5)
        cv2.line(img, self.dict[14], self.dict[16], (0, 0, 255), 5)

    def getVector(self, frame1, frame2):
        return [frame2[0] - frame1[0], frame2[1] - frame1[1]]

    def getElbowAngle(self, img):
        #print("img ", img) # img not None
        self.dict = self.getLandmarkPoint(img)

        #print(self.dict) # Don't know why "return None"
        p1 = self.dict[self.RIGHT_SHOULDER]
        p2 = self.dict[self.RIGHT_ELBOW]
        p3 = self.dict[self.RIGHT_WRIST]

        v1 = self.getVector(p1, p2)
        v2 = self.getVector(p3, p2)

        angle = math.acos(np.dot(
            v1, v2) / math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]) / math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]))
        angle = math.degrees(angle)
        return angle

    def getBodyMovement(self, img1, img2):
        dict1 = self.getLandmarkPoint(img1)
        dict2 = self.getLandmarkPoint(img2)
        diff = 0
        for i in range(33):
            (x1, y1) = dict1[i]
            (x2, y2) = dict2[i]
            diff += (x1 - x2)**2 + (y1 - y2)**2
        return diff

    def getHitPointFrame(self, df):
        q = []
        hitPoints = []
        # first = None
        # max_dist = 0
        for i in range(len(df)):
            if df['Visibility'][i] == 0:
                continue
            # # update x position
            # if int(df.loc[[i]]['X']) >
            # if first == None:
            #     first = int(df.loc[[i]]['X'])
            # else:
            #     dist = abs( int(df.loc[[i]]['X']) - first )
            #     if dist > max_dist:
            #         hitPoints.append(i)
            #         max_dist = dist


            q.append(df.loc[[i]])
            if len(q) == 5:
                # x 方向反轉
                x0 = int(q[0]['X'].values)
                x1 = int(q[1]['X'].values)
                x2 = int(q[2]['X'].values)
                x3 = int(q[3]['X'].values)
                x4 = int(q[4]['X'].values)

                v1 = x1 - x0
                v2 = x2 - x1
                v3 = x3 - x2
                v4 = x4 - x3
                if v1*v2 > 0 and v2*v3 < 0 and v3*v4 > 0:
                    hitPoints.append((x3,i-2)) # 中點q[3]的i,x
                # p0 = [int(q[0]['X'].values), int(q[0]['Y'].values)]
                # v1 = self.getVector(p0, p1)
                # ret = np.dot(v1, v2)
                # if int(ret) < 0:
                #     hitPoint = i-1
                #     break
                q.pop(0)

        hitpoint = -1
        if len(hitPoints) == 1:
            hitpoint = hitPoints[0][1]
        else:
            mid_x = self.w/2
            min_dist = 999999999
            for point in hitPoints:
                (x, i) = point
                if abs(x- mid_x) < min_dist:
                    hitpoint = i
                    min_dist = abs(x- mid_x)

        if hitpoint != -1:
            self.debugLog(f"hitPoint at {hitpoint} frame.\n")
        else:
            self.debugLog("找不到擊球點\n")
            fakeHitPoint = 139
            self.debugLog(f"測試假擊球點 {fakeHitPoint} frame.")
            return fakeHitPoint
        return hitpoint

    def getbackSwingPointFrame(self, hitPoint):
        elbowAngle_hitpoint = self.elbowAngleList[:hitPoint]
        minAngle = min(elbowAngle_hitpoint)
        backSwingPoint = elbowAngle_hitpoint.index(minAngle)
        self.debugLog(f"backSwingPoint at {backSwingPoint} frame.")
        self.debugLog(f"min angle = {minAngle}\n")
        return backSwingPoint

    def getPreparePointFrame(self, cap, backSwingPoint):
        preparePoint = -1
        minMovement = 99999999
        for i in range(backSwingPoint):
            cap.set(1, i)
            ret, frame1 = cap.read()
            ret, frame2 = cap.read()
            tmp = self.getBodyMovement(frame1, frame2)
            if tmp <= minMovement:
                minMovement = tmp
                preparePoint = i
        if preparePoint == -1:
            self.debugLog(f"Cannot detect preparepoint")
        else:
            self.debugLog(f"preparePoint at {preparePoint} frame")
            self.debugLog(f"minMovement = {minMovement}")
        return preparePoint

    def cropImage(self, frame, df_action):
        [minX, minY] = df_action.min(axis=1)
        [maxX, maxY] = df_action.max(axis=1)
        minX = max(0, minX-100)
        maxX = min(frame.shape[1], maxX+100)
        minY = max(0, minY-100)
        maxY = min(frame.shape[0], maxY+100)
        cropFrame = frame[minY:maxY, minX:maxX]
        return cropFrame

    def writeResults(self, cap, hitPoint: int, backSwingPoint: int, preparePoint: int):
        # hit frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, hitPoint)
        ret, frame = cap.read()
        if ret:
            jointPosDict = self.getLandmarkPoint(frame, drawCircle=False)
            df_hitPoint = pd.DataFrame.from_dict(jointPosDict)
            cropFrame = self.cropImage(frame, df_hitPoint)
            cv2.imwrite(os.path.join(self.output_folder, "strike.jpg"), cropFrame)

        # backSwingPoint frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, backSwingPoint)
        ret, frame = cap.read()
        if ret:
            jointPosDict = self.getLandmarkPoint(frame, drawCircle=False)
            df_backSwingPoint = pd.DataFrame.from_dict(jointPosDict)
            cropFrame = self.cropImage(frame, df_backSwingPoint)
            cv2.imwrite(os.path.join(self.output_folder, "ready.jpg"), cropFrame)

        # preparePoint frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, preparePoint)
        ret, frame = cap.read()
        if ret:
            jointPosDict = self.getLandmarkPoint(frame, drawCircle=False)
            df_preparePoint = pd.DataFrame.from_dict(jointPosDict)
            cropFrame = self.cropImage(frame, df_preparePoint)
            cv2.imwrite(os.path.join(self.output_folder, "start.jpg"), cropFrame)

        combine = [df_hitPoint, df_backSwingPoint, df_preparePoint]
        df = pd.concat(combine)
        hitPosX = df.iloc[0]
        hitPosY = df.iloc[1]
        self.hitPosList = np.array([(x, y) for x, y in zip(hitPosX, hitPosY)])
        swingPosX = df.iloc[2]
        swingPosY = df_backSwingPoint.iloc[1]
        self.swingPosList = np.array([(x, y) for x, y in zip(swingPosX, swingPosY)])
        PrepPosX = df_preparePoint.iloc[0]
        PrepPosY = df_preparePoint.iloc[1]
        self.PrepPosList = np.array([(x, y) for x, y in zip(PrepPosX, PrepPosY)])
        return

    # return total analyse result
    def analyse(self):
        self.prepareAnalyzer()
        self.backswingAnalyzer()
        self.hitAnalyzer()
        output_path = os.path.join(self.output_folder, "analyzeResult.json")
        self.outputResult(self.result, output_path)


    # output result to a csv file
    def outputResult(self, result, filePath):
        with open(filePath, "w", encoding="utf8") as f:
            json.dump(result, f, indent = 4, ensure_ascii=False)


    # return prepare action analysis result
    def prepareAnalyzer(self):
        pos = self.PrepPosList
        #1 身體前傾 : 脊椎與地面法向量的角度 >= 7
        neck = (pos[self.LEFT_SHOULDER] + pos[self.RIGHT_SHOULDER])/2
        hip = (pos[self.LEFT_HIP] + pos[self.RIGHT_HIP])/2
        ankle = (pos[self.LEFT_ANKLE] + pos[self.RIGHT_ANKLE])/2

        upperBody = neck - hip
        lowerBody = hip - ankle
        angle = self.getAngle(upperBody, lowerBody)

        bodyLean_pass = True if angle >= 7 else False
        bodyLean_score = 100 if angle >= 7 else 0
        bodyLean_text = f"身體前傾:"
        bodyLean_value = dict()
        bodyLean_value["angle"] = dict()
        bodyLean_value["angle"]["text"] = "身體前傾角度:"
        bodyLean_value["angle"]["value"] = angle

        #2 雙手自然放於身前 : 手腕高度介於肩膀~臀部之間
        if pos[self.LEFT_WRIST][1] >= pos[self.LEFT_SHOULDER][1] and pos[self.LEFT_WRIST][1] <= pos[self.LEFT_HIP][1]:
            if pos[self.RIGHT_WRIST][1] >= pos[self.RIGHT_SHOULDER][1] and pos[self.RIGHT_WRIST][1] <= pos[self.RIGHT_HIP][1]:
                handPosition_pass = True
            else:
                handPosition_pass = False
        else:
            handPosition_pass = '不通過'

        handPosition_text = f"手腕位置:"
        handPosition_score = 100 if handPosition_pass else 0
        handPosition_value = dict()
        handPosition_value["text"] = "手腕高度在臀部肩膀之間"
        handPosition_value["r_shoulder_y"] = dict()
        handPosition_value["r_shoulder_y"]["text"] = "右肩膀高度"
        handPosition_value["r_shoulder_y"]["value"] = int(pos[self.RIGHT_SHOULDER][1])
        handPosition_value["r_wristy"] = dict()
        handPosition_value["r_wristy"]["text"] = "右手腕高度"
        handPosition_value["r_wristy"]["value"] = int(pos[self.RIGHT_WRIST][1])
        handPosition_value["r_hip_y"] = dict()
        handPosition_value["r_hip_y"]["text"] = "右臀部高度"
        handPosition_value["r_hip_y"]["value"] = int(pos[self.RIGHT_HIP][1])
        handPosition_value["l_wrist_y"] = dict()
        handPosition_value["l_wrist_y"]["text"] = "左手腕高度"
        handPosition_value["l_wrist_y"]["value"] = int(pos[self.LEFT_WRIST][1])
        handPosition_value["l_shoulder_y"] = dict()
        handPosition_value["l_shoulder_y"]["text"] = "左肩膀高度"
        handPosition_value["l_shoulder_y"]["value"] = int(pos[self.LEFT_SHOULDER][1])
        handPosition_value["l_hip_y"] = dict()
        handPosition_value["l_hip_y"]["text"] = "左臀部高度"
        handPosition_value["l_hip_y"]["value"] = int(pos[self.LEFT_HIP][1])


        #3 膝蓋微彎 : 大腿與小腿夾角: getRKneeAngle() == getLKneeAngle < 180
        r_thigh_vector = pos[self.RIGHT_HIP] - pos[self.RIGHT_KNEE]
        r_calf_vector = pos[self.RIGHT_ANKLE] - pos[self.RIGHT_KNEE]
        r_kneeAngle = self.getAngle(r_thigh_vector, r_calf_vector)
        l_thigh_vector = pos[self.LEFT_HIP] - pos[self.LEFT_KNEE]
        l_calf_vector = pos[self.LEFT_ANKLE] - pos[self.LEFT_KNEE]
        l_kneeAngle =  self.getAngle(l_thigh_vector, l_calf_vector)

        kneeAngle_pass = True if r_kneeAngle < 180 and l_kneeAngle < 180 else False
        kneeAngle_text = f"膝蓋微彎:"
        kneeAngle_score = 100 if kneeAngle_pass else 0
        kneeAngle_value = dict()
        kneeAngle_value["r_knee_angle"] = dict()
        kneeAngle_value["r_knee_angle"]["text"] = "右膝蓋角度"
        kneeAngle_value["r_knee_angle"]["value"] = r_kneeAngle
        kneeAngle_value["l_knee_angle"] = dict()
        kneeAngle_value["l_knee_angle"]["text"] = "左膝蓋角度"
        kneeAngle_value["l_knee_angle"]["value"] = l_kneeAngle


        #4 雙腳平行 :
        rvector = pos[self.RIGHT_HEEL] - pos[self.RIGHT_HIP]
        lvector = pos[self.LEFT_HEEL] - pos[self.LEFT_HIP]
        angle = self.getAngle(rvector, lvector)

        feetAngle_pass = True if angle < 10 else False
        feetAngle_text = f"雙腳平行:"
        feetAngle_score = 100 if feetAngle_pass else 0
        feetAngle_value = dict()
        feetAngle_value["angle"] = dict()
        feetAngle_value["angle"]["text"] = "雙腳角度"
        feetAngle_value["angle"]["value"] = angle

        # Generate result

        self.result["tstart"] = dict()
        self.result["tstart"]["frame"] = self.preparePoint
        self.result["tstart"]["score"] = dict()
        self.result["tstart"]["score"]["text"] = "總評:"
        self.result["tstart"]["score"]["value"] = (bodyLean_score + handPosition_score + kneeAngle_score + feetAngle_score) / 4

        self.result["tstart"]["body_lean"] = dict()
        self.result["tstart"]["body_lean"]["text"] = bodyLean_text
        self.result["tstart"]["body_lean"]["pass"] = bodyLean_pass
        self.result["tstart"]["body_lean"]["score"] = bodyLean_score
        self.result["tstart"]["body_lean"]["value"] = bodyLean_value

        self.result["tstart"]["hand_position"] = dict()
        self.result["tstart"]["hand_position"]["text"] = handPosition_text
        self.result["tstart"]["hand_position"]["pass"] = handPosition_pass
        self.result["tstart"]["hand_position"]["score"] = handPosition_score
        self.result["tstart"]["hand_position"]["value"] = handPosition_value

        self.result["tstart"]["knee_angle"] = dict()
        self.result["tstart"]["knee_angle"]["text"] = kneeAngle_text
        self.result["tstart"]["knee_angle"]["pass"] = kneeAngle_pass
        self.result["tstart"]["knee_angle"]["score"] = kneeAngle_score
        self.result["tstart"]["knee_angle"]["value"] = kneeAngle_value

        self.result["tstart"]["feet_distance"] = dict()
        self.result["tstart"]["feet_distance"]["text"] = feetAngle_text
        self.result["tstart"]["feet_distance"]["pass"] = feetAngle_pass
        self.result["tstart"]["feet_distance"]["score"] = feetAngle_score
        self.result["tstart"]["feet_distance"]["value"] = feetAngle_value


    # return backswing action analysis result
    def backswingAnalyzer(self):
        pos = self.swingPosList
        #1 右手臂朝前: 右手軸在右膝前方
        if pos[self.RIGHT_ELBOW][0] > pos[self.RIGHT_SHOULDER][0]: # 人朝右
            if pos[self.RIGHT_ELBOW][0] >= pos[self.RIGHT_KNEE][0]:
                handPosition_pass = True
            else:
                handPosition_pass = False
        else:
            if pos[self.RIGHT_ELBOW][0] > pos[self.RIGHT_KNEE][0]: # 人朝左
                handPosition_pass = False
            else:
                handPosition_pass = True

        handPosition_text = f"右手臂朝前:"
        handPosition_score = 100 if handPosition_pass else 0
        handPosition_value = dict()
        handPosition_value["r_elbow_x"] = dict()
        handPosition_value["r_elbow_x"]['text'] = '右手腕位置'
        handPosition_value["r_elbow_x"]['value'] = int(pos[self.RIGHT_ELBOW][0])
        handPosition_value["r_knee_x"] = dict()
        handPosition_value["r_knee_x"]['text'] = '右膝蓋位置'
        handPosition_value["r_knee_x"]['value'] = int(pos[self.RIGHT_KNEE][0])

        # Hitpoint需要的data
        neck = (pos[self.LEFT_SHOULDER] + pos[self.RIGHT_SHOULDER])/2
        hip = (pos[self.LEFT_HIP] + pos[self.RIGHT_HIP])/2
        self.backSwingSpineVector = neck - hip
        self.swingBackArmVector = pos[self.RIGHT_SHOULDER] - pos[self.RIGHT_ELBOW]

        #2 右手臂小臂與大臂約成80~90度: getRElbowAngle() >= 80 && getRElbowAngle <= 90
        self.foreArmVector = pos[self.RIGHT_WRIST] - pos[self.RIGHT_ELBOW]
        self.armAngle = self.getAngle(self.foreArmVector, self.swingBackArmVector)
        elbowAngle_pass = True if self.armAngle >= 80 and self.armAngle <= 90 else False
        elbowAngle_text = f"持拍手肘角度:"
        elbowAngle_score = 100 if elbowAngle_pass else 0
        elbowAngle_value = dict()
        elbowAngle_value["elbow_angle"] = dict()
        elbowAngle_value["elbow_angle"]['text'] = "手肘角度"
        elbowAngle_value["elbow_angle"]['value'] = self.armAngle


        #3 左手臂保持平衡: 前手臂舉起，用前手臂與後手臂的角度判斷有沒有舉起
        foreArmVector_left = pos[self.LEFT_WRIST] - pos[self.LEFT_ELBOW]
        backArmVector_left =pos[self.LEFT_SHOULDER] -  pos[self.LEFT_ELBOW]
        armAngle = self.getAngle(foreArmVector_left, backArmVector_left)
        armBalance_pass = True if armAngle > 25 else False
        armBalance_text = f"非持拍手平衡:"
        armBalance_score = 100 if armBalance_pass else 0
        armBalance_value = dict()
        armBalance_value["arm_angle"] = dict()
        armBalance_value["arm_angle"]["text"] = "手肘角度"
        armBalance_value["arm_angle"]['value'] = armAngle

        #4 右腳在前方: (右腳尖到右腳跟)和(右腳尖到左腳根)的夾角
        footDirection = pos[self.RIGHT_FOOT_INDEX][0] - pos[self.RIGHT_HEEL][0]
        footVector = pos[self.RIGHT_FOOT_INDEX][0] - pos[self.LEFT_FOOT_INDEX][0]
        footPosition_pass = True if footDirection*footVector >= 0 else False
        footPosition_text = f"同側腳在前:"
        footPosition_score = 100 if footPosition_pass else 0

        # save result

        self.result["tready"] = dict()
        self.result["tready"]["frame"] = self.backSwingPoint
        self.result["tready"]["score"] = dict()
        self.result["tready"]["score"]["text"] = "總評:"
        self.result["tready"]["score"]["value"] = (handPosition_score + elbowAngle_score + armBalance_score + footPosition_score) / 4


        self.result["tready"]["hand_position"] = dict()
        self.result["tready"]["hand_position"]["text"] = handPosition_text
        self.result["tready"]["hand_position"]["pass"] = handPosition_pass
        self.result["tready"]["hand_position"]["score"] = handPosition_score
        self.result["tready"]["hand_position"]["value"] = handPosition_value

        self.result["tready"]["elbow_angle"] = dict()
        self.result["tready"]["elbow_angle"]["text"] = elbowAngle_text
        self.result["tready"]["elbow_angle"]["pass"] = elbowAngle_pass
        self.result["tready"]["elbow_angle"]["score"] = elbowAngle_score
        self.result["tready"]["elbow_angle"]["value"] = elbowAngle_value


        self.result["tready"]["arm_balance"] = dict()
        self.result["tready"]["arm_balance"]["text"] = armBalance_text
        self.result["tready"]["arm_balance"]["pass"] = armBalance_pass
        self.result["tready"]["arm_balance"]["score"] = armBalance_score
        self.result["tready"]["arm_balance"]["value"] = armBalance_value


        self.result["tready"]["foot_position"] = dict()
        self.result["tready"]["foot_position"]["text"] = footPosition_text
        self.result["tready"]["foot_position"]["pass"] = footPosition_pass
        self.result["tready"]["foot_position"]["score"] = footPosition_score

    # return hit action analysis result
    def hitAnalyzer(self):
        pos = self.hitPosList
        # 右手臂從架拍到擊球，與脊椎夾角變化<10°
        swingAngle = self.getAngle(self.backSwingSpineVector, self.swingBackArmVector)
        backArmVector = pos[self.RIGHT_ELBOW] - pos[self.RIGHT_SHOULDER]
        spineVector = pos[self.RIGHT_HIP] - pos[self.RIGHT_SHOULDER]
        hitangle = self.getAngle(backArmVector, spineVector)
        angleDiff = np.abs(hitangle - swingAngle)
        elbowFix_pass = True if angleDiff <= 20 else False
        elbowFix_text = f"右手臂小幅變化:"
        elbowFix_score = 100 if elbowFix_pass else 0

        # 球的軌跡水平飛出: 出球角度<15° 用之前
        # 需要軌跡資料

        foreArmVector = pos[self.RIGHT_WRIST] - pos[self.RIGHT_ELBOW]
        backArmVector = pos[self.RIGHT_SHOULDER] - pos[self.RIGHT_ELBOW]
        armAngle = self.getAngle(foreArmVector, backArmVector)
        hitAngle_pass = True if armAngle >= 90 else False
        hitAngle_text = f"擊球手肘角度:"
        print(f"hit angle {armAngle}")
        hitAngle_score = 100 if hitAngle_pass else 0


        self.result["tstrike"] = dict()
        self.result["tstrike"]["frame"] = self.hitPoint
        self.result["tstrike"]["score"] = dict()
        self.result["tstrike"]["score"]["text"] = "總評"
        self.result["tstrike"]["score"]["value"] = (elbowFix_score + hitAngle_score)/2

        self.result["tstrike"]["elbow_fix"] = dict()
        self.result["tstrike"]["elbow_fix"]["text"] = elbowFix_text
        self.result["tstrike"]["elbow_fix"]["pass"] = elbowFix_pass
        self.result["tstrike"]["elbow_fix"]["score"] = elbowFix_score
        self.result["tstrike"]["elbow_fix"]["value"] = dict()
        self.result["tstrike"]["elbow_fix"]["value"]["angle_diff"] = dict()
        self.result["tstrike"]["elbow_fix"]["value"]["angle_diff"]['text'] = "右手夾角變化"
        self.result["tstrike"]["elbow_fix"]["value"]["angle_diff"]['value'] = angleDiff

        self.result["tstrike"]["hit_angle"] = dict()
        self.result["tstrike"]["hit_angle"]["text"] = hitAngle_text
        self.result["tstrike"]["hit_angle"]["pass"] = hitAngle_pass
        self.result["tstrike"]["hit_angle"]["score"] = hitAngle_score
        self.result["tstrike"]["hit_angle"]["value"] = dict()
        self.result["tstrike"]["hit_angle"]["value"]["arm_angle"] = dict()
        self.result["tstrike"]["hit_angle"]["value"]["arm_angle"]['text'] = "手肘角度"
        self.result["tstrike"]["hit_angle"]["value"]["arm_angle"]['value'] = armAngle


    def getAngle(self, vectorA, vectorB):
        cosine = np.dot(vectorA, vectorB)/norm(vectorA)/norm(vectorB)
        angle = np.arccos(np.clip(cosine, -1, 1))
        angle = np.degrees(angle)
        return angle

    def debugLog(self, log):
        # if self.debugging == True:
        print(log)

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'Flat Ball Analyzer')
    parser.add_argument('--input_video', type=str, default = 'CameraReaderL.avi', help='Input video path')
    parser.add_argument('--input_csv', type=str, default = 'TrackNetL.csv', help='Input csv path')
    parser.add_argument('--output_folder', type=str, default = '.', help='Analyzation result will be at [output_folder]/[input_video]_Analyze')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    tracker = FlatballActionAnalyzer(videoPath = args.input_video, trackNetCSV = args.input_csv, output_folder = args.output_folder)
    return


if __name__ == "__main__":
    main()
