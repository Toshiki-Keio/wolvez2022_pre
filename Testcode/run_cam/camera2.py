import os
import numpy as np
import cv2
import time

current_dir=os.getcwd()
frame_rate=20
size=(1280,720)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
video = cv2.VideoWriter(current_dir+'/Testcode/run_cam/images/movie.mp4',
                      fourcc, frame_rate, size)
cap=cv2.VideoCapture(0)


start=time.time()
print(current_dir)
while True:
    ret, frame=cap.read()
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
time_needed=time.time()-start
print(time_needed)
cap.release()
cv2.destroyAllWindows()
