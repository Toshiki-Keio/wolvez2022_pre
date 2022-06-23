import RPi.GPIO as GPIO
import sys
import cv2
sys.path.append("/home/pi/Desktop/wolvez2021/Testcode/sensor_integration/LoRa_SOFT")
import time
import numpy as np
import os
from bno055 import BNO055
from motor import motor
import constant as ct
import gps

class Cansat():
    def __init__(self,state):
        # GPIO設定
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM) #GPIOの設定
        GPIO.setup(ct.const.FLIGHTPIN_PIN,GPIO.IN) #フライトピン用
        GPIO.setup(ct.const.SEPARATION_PIN,GPIO.OUT) #焼き切り用のピンの設定
        
        # インスタンス生成        
        self.bno055 = BNO055()
        self.bno055.setupBno()
        self.rightMotor = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
        self.leftMotor = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)
        self.gps = gps.GPS()
        self.cap = cv2.VideoCapture(0)
        
        #初期パラメータ設定
        self.timer = 0
        self.state = state
                #stateに入っている時刻の初期化
        self.startTime = time.time()
        self.preparingTime = 0
        self.flyingTime = 0
        self.droppingTime = 0
        # self.landingTime = 0
        # self.pre_motorTime = 0
        # self.startingTime = 0
        # self.measureringTime = 0
        # self.runningTime = 0
        # self.positioningTime = 0
        # self.finishTime = 0
        
        #state管理用変数初期化
        self.gpscount=0
        self.startgps_lon=[]
        self.startgps_lat=[]
        
        self.countPreLoop = 0
        self.countFlyLoop = 0
        self.countDropLoop = 0
        self.countSwitchLoop=0
        self.countGoal = 0
        self.countgrass=0
        
    
    def writeData(self):
        #ログデータ作成。\マークを入れることで改行してもコードを続けて書くことができる
        print_datalog = str(self.timer) + ","\
                  + "state:"+str(self.state)+ ","\
                  + "Lat:"+str(self.gps.Lat).rjust(6) + ","\
                  + "Lng:"+str(self.gps.Lon).rjust(6) + ","\
                  + "rV:" + str(round(self.rightMotor.velocity,2)).rjust(6) + ","\
                  + "lV:" + str(round(self.leftMotor.velocity,2)).rjust(6) + ","\
                  + "q:" + str(self.bno055.ex).rjust(6) 
        print(print_datalog)
        
        datalog = str(self.timer) + ","\
                  + str(self.state) + ","\
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

    def sequence(self):
        if self.state == 0:
            self.preparing()
        elif self.state == 1:
            self.flying()
        elif self.state == 2:
            self.dropping()
        # elif self.state == 3:
        #     self.run_motor()
        # elif self.state == 4:
        #     self.starting()
        # elif self.state == 5:
        #     self.measuring()
        # elif self.state == 6:
        #     self.running()
        # elif self.state == 7:
        #     self.positioning()
        # elif self.state == 8:
        #     self.finish()
        else:
            self.state = self.laststate #どこにも引っかからない場合何かがおかしいのでlaststateに戻してあげる
            
    def sensor(self):#セットアップ終了後
        self.timer = int(1000*(time.time() - self.startTime)) #経過時間 (ms)
        self.gps.gpsread()
        self.bno055.bnoread()
        self.ax=round(self.bno055.ax,3)
        self.ay=round(self.bno055.ay,3)
        self.az=round(self.bno055.az,3)
        self.ex=round(self.bno055.ex,3)
        
        self.writeData()#txtファイルへのログの保存
    
#         if not self.state == 1: #preparingのときは電波を発しない
#             if not self.state ==5:#self.sendRadio()#LoRaでログを送信
#                 self.sendRadio()
#             else:
#                 self.rightMotor.stop()
#                 self.leftMotor.stop()
#                 self.switchRadio()

    def preparing(self):#時間が立ったら移行
        if self.preparingTime == 0:
            self.preparingTime = time.time()#時刻を取得
            self.rightMotor.stop()
            self.leftMotor.stop()
        #self.countPreLoop+ = 1
        if not self.preparingTime == 0:
            if self.gpscount <= ct.const.PREPARING_GPS_COUNT_THRE:
                self.startgps_lon.append(float(self.gps.Lon))
                self.startgps_lat.append(float(self.gps.Lat))
                self.gpscount+=1
                
            else:
                print("GPS completed!!")
            
            if time.time() - self.preparingTime > ct.const.PREPARING_TIME_THRE:
                self.startlon=np.mean(self.startgps_lon)
                self.startlat=np.mean(self.startgps_lat)
                self.state = 1
                self.laststate = 1
    
    def flying(self):#フライトピンが外れたのを検知して次の状態へ以降
        if self.flyingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.flyingTime = time.time()

        if GPIO.input(ct.const.FLIGHTPIN_PIN) == GPIO.HIGH:#highかどうか＝フライトピンが外れているかチェック
            self.countFlyLoop+=1
            if self.countFlyLoop > ct.const.FLYING_FLIGHTPIN_COUNT_THRE:#一定時間HIGHだったらステート移行
                self.state = 2
                self.laststate = 2       
        else:
            self.countFlyLoop = 0 #何故かLOWだったときカウントをリセット
    
    def dropping(self):
        if self.droppingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.droppingTime = time.time()
      
        #加速度が小さくなったら着地判定
        if (pow(self.bno055.ax,2) + pow(self.bno055.ay,2) + pow(self.bno055.az,2)) < pow(ct.const.DROPPING_ACC_THRE,2):#加速度が閾値以下で着地判定
            self.countDropLoop+=1
            self.separation()
            
            if self.countDropLoop > ct.const.DROPPING_ACC_COUNT_THRE:
                self.state = 3
                self.laststate = 3
        else:
            self.countDropLoop = 0 #初期化の必要あり

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
        magnet="mx="+str(self.bno055.mx)+","\
              +"my="+str(self.bno055.my)+","\
              +"mz="+str(self.bno055.mz)
        print(grav,euler,magnet) 
        
    def separation(self):
        GPIO.output(ct.const.SEPARATION_PIN,1) #電圧をHIGHにして焼き切りを行う
        time.sleep(ct.const.SEPARATION_TIME) #継続時間を指定
        GPIO.output(ct.const.SEPARATION_PIN,0) #電圧をLOWにして焼き切りを終了する
        
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