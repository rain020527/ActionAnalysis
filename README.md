# ActionAnalysis
Sport action analysis can enable athletes to do more correct actions, but it takes too long to develop algorithms for each action. The action analysis framework is proposed to systematically evaluate the different types of sports actions. The entire framework consists of four structured modules, including data extraction, key frame detection, posture evaluation, and movement evaluation. Besides, integrate many common action features and encapsulate into functions to reduce development time. The framework is implemented on an application named CoachBox to evaluate the learner's action and increase sports learning effectiveness. This device can automatically analyze the action through deep learning without any sensor. In addition, a coordinate transform method is provided to convert skeleton coordinates to real-world coordinates for visualization.

<img src="https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/system_framework.png" width="80%"/>

## Resource Link
### Demo Video
[FUTEX2021 - The clip of TV interview](https://www.youtube.com/watch?v=3iLHowULkU8&ab_channel=USTV%E9%9D%9E%E5%87%A1%E9%9B%BB%E8%A6%96)

[FUTEX2021 - Introduction of CoachBox](https://www.youtube.com/watch?v=nyMEjmVBgJs&ab_channel=%E6%9C%AA%E4%BE%86%E7%A7%91%E6%8A%80%E9%A4%A8FUTEX-FutureTech)

[FUTEX2021 - Conference: CoachBox Visualized badminton hitting action analysis plan](https://www.youtube.com/watch?v=Gx80Wrej4nM&ab_channel=%E6%9C%AA%E4%BE%86%E7%A7%91%E6%8A%80%E9%A4%A8FUTEX-FutureTech)

### Github Link
https://github.com/rain020527/ActionAnalysis

### Dataset Link
[TODO]
### Master Thesis
[TODO]

## Contects
1. [Source File Description](#Source-File-Description)
2. [Installation](#Installation)
3. [Demo](#Demo)
4. [Implement Different Action](#Implement-Different-Action)
5. [Court Coordinate System Transform](#Court-Coordinate-System-Transform)
6. [The Result After Coordinate Transformation](#The-Result-After-Coordinate-Transformation)

## Source File Description
1. "ActionAnalyst" folder includes different types of action analysis
2. "ActionAnalyst/lib" folder is the core codes that do the skeleton detection (PklGenerator.py) and construct the analysis base class (VibePklParser). 
3. The folder "replay" includes the data we collect by CoachBox or another webcam. If you want to use another webcam data, , you need to prepare the camera parameters, the csv file of the ball trajectory to execute the demo correctly.
4.  The folder "lib" and "VIBE" is a library that provide the function to process the camera config and skeleton detection.

## Installation
The test environment is ubuntu 18.04, python3.8 version.
1. Install the VIBE: you can refer to the original VIBE github to download. https://github.com/mkocabas/VIBE
```
git clone https://github.com/mkocabas/VIBE.git

source scripts/install_pip.sh

source scripts/prepare_data.sh
```

2. Install the other library:
```
pip3 install joblib

pip3 install numpy

pip3 install opencv-python

pip3 install jsons

pip3 install pandas

pip3 install mysqlclient
```

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

- Key Frame Detection, Posture and Movement Evaluation

In a series of actions, there are usually several time points that are representative. This study tries to identify these key frames automatically, and evaluate the posture and movement to measure whether the action is standard.

<img src="https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/Smash_framework.png" width="75%"/>

- Analysis Result(json file)
                                                                                                           
Can see the analysis result, such as the right knee angle is 148 degrees and the left knee angle is 163 degrees. Feet width vs shoulder width that can evaluate whether the start posture is wrong.
                                                                                                           
<img src="https://github.com/rain020527/ActionAnalysis/blob/main/readme_img/report.png" width="75%"/>


## Implement Different Action
This project just only complete the evaluation algorithm of badminton backcourt action. If you want to develop the other action evaluation algorithm or the key frame detection by yourself, you can refer to the high ball folder. Such as high ball, you can implement the action analysis algorithm by inheriting class VibePklParser. You need to implement two virtual function: 
1. time_slice(), that is your key frame detection algorithm, and return the tuple of the key frame number.
2. run(), that is the evaluation algorithm to analyze the postures and movements. You can also refer to the ActionAnalyst/High folder that we provide to implement the action analysis you want.
Modify the main.py running evaluation algorithm to your algorithm and change the threshold you need. Then you can execute and see the result in the json file.
If you want to complete the analysis of other actions, such as rehabilitation or various ball games, you can also develop it yourself through this framework.
