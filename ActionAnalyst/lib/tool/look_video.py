import cv2
import os
import csv
import argparse
import sys
import pandas as pd

def toInt(df):
    df.Frame = df.Frame.astype('int64')
    df.Visibility = df.Visibility.astype('int64')
    df.Event = df.Event.astype('int64')
    return df

videofile = sys.argv[1]
videofile_basename = os.path.splitext(os.path.basename(videofile))[0]
csvfile = os.path.splitext(videofile)[0] + '.csv'

use_csv = False

OUTPUT_FOLDER = os.path.join('./output/', videofile_basename)
#if not os.path.exists(OUTPUT_FOLDER):
#    os.makedirs(OUTPUT_FOLDER)

video = cv2.VideoCapture(videofile)
total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print (f"Total frame: {total_frame}")

if os.path.isfile(csvfile):
    use_csv = True
    label_csv = []
    with open(csvfile, newline='') as c_f:
        rows = csv.reader(c_f)
        for index, row in enumerate(rows):
            if index == 0:
                continue
            label_csv.append(row)
    #assert len(label_csv)==total_frame, 'Video Total Frame != CSV Total Frame, Please Report!'

fps = float(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"FPS: {fps} Width: {output_width}, Height: {output_height}")
fourcc = cv2.VideoWriter_fourcc(*'XVID')

currentFrame = 0
print(f"Current frame: {currentFrame}")

images = [None] * total_frame

i = 0
while(True):
    success, image = video.read()
    if not success:
        break
    images[i] = image
    i = i+1

image=images[currentFrame]
cv2.namedWindow("image")


output_video_idx = 1


start = None
end = None

N = 100

while(True):
    cv2.imshow("image", cv2.resize(image,(output_width*2//3,output_height*2//3)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord("x"):     #jump next frame
        if currentFrame < total_frame-1:
            if images[currentFrame+1] is not None:
                image=images[currentFrame+1]
                currentFrame+=1
                print('Current frame: ', currentFrame)
            else:
                print(f'Frame {currentFrame+1} is broken')
        else:
            print('This is the last frame')
    elif key == ord("b"):     #jump next N frame
        if currentFrame < total_frame-N:
            if images[currentFrame+N] is not None:
                image=images[currentFrame+N]
                currentFrame+=N
                print('Current frame: ', currentFrame)
            else:
                print(f'Frame {currentFrame+N} is broken')
        else:
            print(f'This is the last {N} frame')
    elif key == ord("z"):     #jump last frame
        if currentFrame == 0:
            print('\nThis is the first frame')
        else:
            currentFrame-=1
            print('Current frame: ', currentFrame)
            image=images[currentFrame]
    elif key == ord("v"):     #jump last N frame
        if currentFrame < N:
            print(f'This is the first {N} frame')
        else:
            currentFrame-=N
            print('Current frame: ', currentFrame)
            image=images[currentFrame]
    elif key == ord("s"):
        if start is None:
            start = currentFrame
            print(f"START AT {currentFrame}")
        else:
            print("YOU DIDNT PRESS E AFTER last S !")
            sys.exit(1)
    elif key == ord("e"):
        end = currentFrame
        if start is None:
            print("YOU FORGET TO PRESS S !")
            sys.exit(1)
        elif start > end:
            print("THE FRAME OF START > END !")
            sys.exit(1)
        else:
            save = True
            if use_csv:
                tmp = []
                for i in range(start,end+1):
                    tmp.append(label_csv[i])
                    if i != start and int(label_csv[i][0]) != int(label_csv[i-1][0])+1:
                        print(f"FRAME {start} TO {end} Drop Frame! NO SAVE!")
                        save=False
                if save:
                    output_csv = f'{videofile_basename}_{output_video_idx}.csv'
                    output_csv_path = os.path.join(OUTPUT_FOLDER,output_csv)
                    print(f"SAVE CSV {start} TO {end} into {output_csv}")
                    pd_df = pd.DataFrame(tmp, columns=['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp']) 
                    pd_df = toInt(pd_df)
                    pd_df.to_csv(output_csv_path, encoding = 'utf-8',index = False)
            if save:
                # Output start~end to video
                output_filename = f'{videofile_basename}_{output_video_idx}.avi'
                output_video_path = os.path.join(OUTPUT_FOLDER,output_filename)
                print(f"SAVE FRAME {start} TO {end} into {output_filename}")
                output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width,output_height))

                for i in range(start,end+1):
                    frame = images[i]
                    output_video.write(frame)
                output_video.release()
                output_video_idx += 1
                start = None
                end = None
                print("SAVE DONE")
    elif key == ord("c"): #cancel
        print("Clear Start and End")
        start = None
        end = None
    elif key == ord("q"):
        if start or end:
            print("YOU FORGET TO PRESS E !")
        break
    

video.release()
