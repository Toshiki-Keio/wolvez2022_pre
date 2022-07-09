#Last Update 2022/07/02
#Author : Toshiki Fukui

import RPi.GPIO as GPIO
import time
import constant as ct
import cv2
from motor import motor
from bno055 import BNO055

try:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM) #GPIOの設定
    # センサのセットアップ用
    # BNO
    bno055 = BNO055()
    bno055.setupBno()
    if bno055.begin() is not True:
        print("Error initializing device")
        exit()
    
    #Motor
    rightMotor = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
    leftMotor = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)

    #camera
    cap = cv2.VideoCapture(0)

    while True:
        #モータ回転
        rightMotor.go(ct.const.MOTOR_VREF)
        leftMotor.go(ct.const.MOTOR_VREF)
        #BNO
        bno055.bnoread()
        #カメラ
        ret,img = cap.read()
#         cv2.imshow("img",img)
        if ret: # execute only when img obtained
            datalog = "ax:"+str(round(bno055.ax,8)).rjust(6) + ","\
                          + "ay:"+str(round(bno055.ay,8)).rjust(6) + ","\
                          + "az:"+str(round(bno055.az,8)).rjust(6) + ","\
                          + "rV:" + str(round(rightMotor.velocity,2)).rjust(6) + ","\
                          + "lV:" + str(round(leftMotor.velocity,2)).rjust(6) + ","\
                          + "q:" + str(bno055.ex).rjust(6) + ","\
                          + "Camera:" + str(img.shape)
            print(datalog)
#         time.sleep(0.3)
    
except KeyboardInterrupt:
    rightMotor.stop()
    leftMotor.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()