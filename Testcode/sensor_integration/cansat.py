#Last Update 2022/06/23
#Author: Toshiki Fukui

import RPi.GPIO as GPIO
import sys
import cv2
sys.path.append("/home/pi/Desktop/wolvez2021/Testcode/sensor_integration/LoRa_SOFT")
import time
import numpy as np
import os
from bno055 import BNO055
from motor import motor
import gps
import constant as ct

class Cansat():
    def __init__(self):
        #GPIOの設定
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        
        #
        self.bno055 = BNO055()
        self.bno055.setupBno()
        self.rightMotor = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
        self.leftMotor = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)

        self.timer = 0
        self.gps = gps.GPS()
        self.cap = cv2.VideoCapture(0)
        self.startTime = time.time()
    
    def writeData(self):
        #ログデータ作成。\マークを入れることで改行してもコードを続けて書くことができる
        print_datalog = str(self.timer) + ","\
                  + "Lat:"+str(self.gps.Lat).rjust(6) + ","\
                  + "Lng:"+str(self.gps.Lon).rjust(6) + ","\
                  + "rV:" + str(round(self.rightMotor.velocity,2)).rjust(6) + ","\
                  + "lV:" + str(round(self.leftMotor.velocity,2)).rjust(6) + ","\
                  + "q:" + str(self.bno055.ex).rjust(6) 
        print(print_datalog)
        
        datalog = str(self.timer) + ","\
                  + str(self.gps.Lat).rjust(6) + ","\
                  + str(self.gps.Lon).rjust(6) + ","\
                  + str(self.bno055.ax).rjust(6) + ","\
                  + str(self.bno055.ay).rjust(6) + ","\
                  + str(self.bno055.az).rjust(6) + ","\
                  + str(round(self.rightMotor.velocity,3)).rjust(6) + ","\
                  + str(round(self.leftMotor.velocity,3)).rjust(6) + ","\
                  + str(self.bno055.ex).rjust(6) 
        
        with open('test.txt',"a")  as test: # [mode] x:ファイルの新規作成、r:ファイルの読み込み、w:ファイルへの書き込み、a:ファイルへの追記
            test.write(datalog + '\n')

    def setup(self):
        self.gps.setupGps()
        # os.system("sudo insmod LoRa_SOFT/soft_uart.ko")
        self.bno055.setupBno()

        if self.bno055.begin() is not True:
            print("Error initializing device")
            exit()
    
    def run(self):#セットアップ終了後
        self.timer = int(1000*(time.time() - self.startTime)) #経過時間 (ms)
        self.getbno055()#BNO取得
#         self.LoRa.sensor()#GPS取得、LoRa通信？？
        self.run_motor()#モータ走行
        img = self.camera(self.cap)#カメラ撮影
        self.writeData()#txtファイルへのログの保存
        
    def getbno055(self):      
        self.bno055.bnoread()
        self.bno055.ax=round(self.bno055.ax,3)
        self.bno055.ay=round(self.bno055.ay,3)
        self.bno055.az=round(self.bno055.az,3)
        self.bno055.gx=round(self.bno055.gx,3)
        self.bno055.gy=round(self.bno055.gy,3)
        self.bno055.gz=round(self.bno055.gz,3)
        self.bno055.ex=round(self.bno055.ex,3)
        self.bno055.ey=round(self.bno055.ey,3)
        self.bno055.ez=round(self.bno055.ez,3)
        accel="ax="+str(self.bno055.ax)+","\
              +"ay="+str(self.bno055.ay)+","\
              +"az="+str(self.bno055.az)
        grav="gx="+str(self.bno055.gx)+","\
              +"gy="+str(self.bno055.gy)+","\
              +"gz="+str(self.bno055.gz) # including gravity
        euler="ex="+str(self.bno055.ex)+","\
              +"ey="+str(self.bno055.ey)+","\
              +"ez="+str(self.bno055.ez)
#         print(accel,euler) 
                  
    def run_motor(self):
        self.rightMotor.go(ct.const.MOTOR_VREF)
        self.leftMotor.go(ct.const.MOTOR_VREF)
    
    def camera(self,cap):
        ret, img = cap.read()      
        cv2.imshow('camera', img)
        return img

    def keyboardinterrupt(self):
        self.rightMotor.stop()
        self.leftMotor.stop()
        self.cap.release()
        cv2.destroyAllWindows()