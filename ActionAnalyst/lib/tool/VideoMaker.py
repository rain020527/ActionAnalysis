# -*- coding:utf-8 -*-
import cv2
import sys


# 來源影片名稱
INPUT_FILE = sys.argv[1] 

# 欲輸出之影片名稱
filename = sys.argv[1].split('.')[0] + '_'

hit_frame_list = []
for i in range(2,len(sys.argv)):
    print(sys.argv[i])
    hit_frame_list.append(int(sys.argv[i]))

# cv2的影片輸出編碼方式
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 抓影片的資訊
reader = cv2.VideoCapture(INPUT_FILE)

# 欲輸出影片的寬
output_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

# 欲輸出影片的高
output_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 輸出影片的FPS (影片每秒有多少張相片)
fps = int(reader.get(cv2.CAP_PROP_FPS))


print('Open Success :',reader.isOpened())
print('origin avi fps :',fps)

start_frame_list = [] # 開始的時間區段 以幀數為單位
end_frame_list = [] # 結束的時間區段 以幀數為單位
for i in range(len(hit_frame_list)):
    start_frame_list.append(int(hit_frame_list[i] - fps * 1.5))
    end_frame_list.append(int(hit_frame_list[i] + fps * 1.5))

print(hit_frame_list)

writer = []
video_frame_list = []
for i in range(2,len(sys.argv)):
    writer.append(cv2.VideoWriter(filename + sys.argv[i] + '.avi', fourcc,fps,(output_width, output_height))) # opencv 寫檔專用的class

vc = cv2.VideoCapture(INPUT_FILE)
c = 0 # 第幾幀
cut = 0 # 紀錄要判斷的時間區間
if vc.isOpened() == True: # 開檔成功
    while True:
        rval, video_frame = vc.read() # 讀出frame
        if rval == True: # 成功讀取frame
            if (c >= start_frame_list[cut]) and (c <= end_frame_list[cut]): # 判斷是否有在時間區間內
                video_frame_list.append(video_frame) 
            c = c + 1
            if ( cut < len(start_frame_list)) and (c > end_frame_list[cut]): # 判斷是否要更新時間區間
                for i in range(len(video_frame_list)):
                    writer[cut].write(video_frame_list[i]) # 將video_frame_list寫入output file
                cut = cut + 1
                video_frame_list = []
            if ( cut >= len(start_frame_list)):
                break
        else:
            break
else:
    print('Video open failed')


# 釋放讀寫的class
vc.release()
for i in range(len(sys.argv)-2):
    writer[i].release()
reader.release()
cv2.destroyAllWindows()