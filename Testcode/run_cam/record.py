import numpy as np
import cv2
import sys
import os
import shutil
import time
import datetime

def camera(save_dir_path,cam_num,fps=1):
    img_dir=save_dir_path
    #img_dir="class_CV/final_report_SfM/img_calib"
    try:
        shutil.rmtree(img_dir)
    except FileNotFoundError:
        pass
    os.mkdir(img_dir)

    ignition=time.time()
    cap1 = cv2.VideoCapture(cam_num)
    print("camera recognized:",cap1.isOpened())
    print("Camera ignitting...")
    while time.time()-ignition<3:
        ret, frame = cap1.read()
        if ret:
            cv2.imwrite(save_dir+"/image.jpg", frame)
    print("camera ready")
    while True:
        ret, frame = cap1.read()
        if ret:
            name=str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")[:16]
            cv2.imwrite(save_dir+f"/{name}.jpg", frame)
            time.sleep(1/fps)

curr_path=os.getcwd()
save_dir=curr_path+"/Testcode/run_cam/images"
camera(save_dir,cam_num=0,fps=1)

