#Last Update 2022/07/02
#Author : Toshiki Fukui

import RPi.GPIO as GPIO
import time
import constant as ct
import cv2
import motor
from bno055 import BNO055

try:
    ##センサのセットアップ用
    #BNO
    self.bno055 = BNO055()
    self.bno055.setupBno()
    if self.bno055.begin() is not True:
        print("Error initializing device")
        exit()
    
    #Motor
    self.rightMotor = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
    self.leftMotor = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)

    #camera
    cap = cv2.VideoCapture(0)

    while True:
        #モータ回転
        self.rightMotor.go(ct.const.MOTOR_VREF)
        self.leftMotor.go(ct.const.MOTOR_VREF)
        #BNO
        self.bno055.bnoread()
        #カメラ
        ret,img = cap.read()
        
        datalog = "ax:"+str(round(self.bno055.ax,8)).rjust(6) + ","\
                      + "ay:"+str(round(self.bno055.ay,8)).rjust(6) + ","\
                      + "az:"+str(round(self.bno055.az,8)).rjust(6) + ","\
                      + "rV:" + str(round(self.rightMotor.velocity,2)).rjust(6) + ","\
                      + "lV:" + str(round(self.leftMotor.velocity,2)).rjust(6) + ","\
                      + "q:" + str(self.bno055.ex).rjust(6) + ","\
                      + "Camera:" + img)
        print(datalog)
    
except KeyboardInterrupt:
    self.rightMotor.stop()
    self.leftMotor.stop()
    cap.release()
    cv2.destroyAllWindows()
