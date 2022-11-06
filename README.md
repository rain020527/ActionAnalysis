# ActionAnalysis

![Image text](https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/system_framework.png)
## Contects
1. [File Description](#File-Description)
2. [Demo](#Demo)
3. [Implement Different Action](#Implement-Different-Action)

## File Description
ActionAnalyst folder includes different types of action analysis, and the ActionAnalyst/lib folder is the core codes that do the skeleton detection (PklGenerator.py) and construct the analysis base class (VibePklParser). The folder replay includes the data we collect by CoachBox or another webcam. If you want to use another webcam data, , you need to prepare the camera parameters, the csv file of the ball trajectory to execute the demo correctly. The folder lib and VIBE is a library that provide the function to process the camera config and skeleton detection.

## Demo
Run the main.py, that includes two modules: 
1. PklGenerator.py that can get the skeletons information of VIBE. 
2. Smash_VibePklParser.py that analyzes the action.
```
$ cd ActionAnalyst/Smash

$ python3 main.py --camera_cfg ../../replay/20220616_134819/4/28124278.cfg --output_folder ../../replay/20220616_134819/4/ --run ../../replay/20220616_134819/4/CameraReaderL.avi --fps 120
```
The result will store in ../../replay/20220616_134819/4/CameraReaderL_Analyze/analyzeResult.json
You can use the CoachBox replay feature to see the analysis result or check the json file.
![Image text](https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/UI.png)

## Implement Different Action
Such as high ball, you can implement the key frame detection algorithm and evalution algorithm by inheriting class VibePklParser. You need to implement two virtual function: 
1. time_slice(), that is your key frame detection algorithm, and return the tuple of the key frame number.
2. run(), that is the evaluation algorithm to analyze the postures and movements. You can also refer to the ActionAnalyst/High folder that we provide to implement the action analysis you want.

## Court Coordinate System Transform
There is a function get3DSKP_court(self, fid, keypoint_idx) in VibePklParser.py. This function output 3D coordinate of pose keypoint in court coordinate space, and you just input the frame index and the keypoint index in self.getJointNames. The principle of this function is as follows: 
![Image text](https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/court_transform.png)
This is the result of court transform. The origin of the court coordinate system is at the center of the stadium, the short axis is X, the long axis is Y, the ground is Z, and 1 unit is 1 meter. 
![Image text](https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/transform_result.png)
