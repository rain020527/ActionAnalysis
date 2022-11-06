# ActionAnalysis
Master's Research
# Action analysis using the skeleton detection and shuttlecock trajectory

## Contects
1. [File Description](#File-Description)
2. [Demo](#Demo)
3. [Implement Different Action](#Implement-Different-Action)

## File Description


## Demo
Run the main.py, that includes two modules: 1. PklGenerator.py that can get the skeletons information of VIBE. 2. Smash_VibePklParser.py that analyzes the action.
```
$ cd ActionAnalyst/Smash

$ python3 main.py --camera_cfg ../../replay/20220616_134819/4/28124278.cfg --output_folder ../../replay/20220616_134819/4/ --run ../../replay/20220616_134819/4/CameraReaderL.avi --fps 120
```
The result will store in ../../replay/20220616_134819/4/CameraReaderL_Analyze/analyzeResult.json
You can use the CoachBox replay feature to see the analysis result or check the json file.

## Implement Different Action
Such as high ball, you can implement the key frame detection algorithm and evalution algorithm by inheriting class VibePklParser. You need to implement two virtual function: 1. time_slice(), that is your key frame detection algorithm, and return the tuple of the key frame number. 2. run(), that is the evaluation algorithm to analyze the postures and movements. You can also refer to the ActionAnalyst/High folder that we provide to implement the action analysis you want.
