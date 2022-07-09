#Last Update 2022/07/02
#Author : Toshiki Fukui

import RPi.GPIO as GPIO
import sys
import cv2
sys.path.append("/home/pi/Desktop/wolvez2021/Testcode/sensor_integration/LoRa_SOFT")
import time
import numpy as np
import os
import re
from datetime import datetime
from glob import glob
from math import prod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from baba_into_window import IntoWindow
from bbaa_learn_dict import LearnDict
from bcaa_eval import EvaluateImg

from bno055 import BNO055
from motor import motor
from gps import GPS
from radio import radio
from led import led
import constant as ct

class Cansat():
    def __init__(self,state):
        # GPIO設定
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM) #GPIOの設定
        GPIO.setup(ct.const.FLIGHTPIN_PIN,GPIO.IN,pull_up_down=GPIO.PUD_UP) #フライトピン用。プルアップを有効化
        GPIO.setup(ct.const.SEPARATION_PIN,GPIO.OUT) #焼き切り用のピンの設定
        
        # インスタンス生成        
        self.bno055 = BNO055()
        self.rightMotor = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
        self.leftMotor = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)
        self.gps = GPS()
        self.radio = radio()
        self.RED_LED = led(ct.const.RED_LED_PIN)
        self.BLUE_LED = led(ct.const.BLUE_LED_PIN)
        self.GREEN_LED = led(ct.const.GREEN_LED_PIN)
        
        #初期パラメータ設定
        self.timer = 0
        self.state = state
        self.startTime = time.time()
        self.preparingTime = 0
        self.flyingTime = 0
        self.droppingTime = 0
        self.landingTime = 0
        self.landstate = 0
        self.firstlearnimgcount = 0
        self.camerastate = 0
        self.camerafirst = 0
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

        self.dict_list = {}
        self.saveDir = "あとで変更"
    
    def writeData(self):
        #ログデータ作成。\マークを入れることで改行してもコードを続けて書くことができる
        print_datalog = str(self.timer) + ","\
                  + "state:"+str(self.state)+ ","\
                  + "Time:"+str(self.gps.Time) + ","\
                  + "Lat:"+str(self.gps.Lat).rjust(6) + ","\
                  + "Lng:"+str(self.gps.Lon).rjust(6) + ","\
                  + "ax:"+str(round(self.bno055.ax,6)).rjust(6) + ","\
                  + "ay:"+str(round(self.bno055.ay,6)).rjust(6) + ","\
                  + "az:"+str(round(self.bno055.az,6)).rjust(6) + ","\
                  + "rV:" + str(round(self.rightMotor.velocity,2)).rjust(6) + ","\
                  + "lV:" + str(round(self.leftMotor.velocity,2)).rjust(6) + ","\
                  + "q:" + str(self.bno055.ex).rjust(6) + ","\
                  + "Camera:" + str(self.camerastate)

        print(print_datalog)
        
        datalog = str(self.timer) + ","\
                  + "state:"+str(self.state) + ","\
                  + "Time:"+str(self.gps.Time) + ","\
                  + "Lat:"+str(self.gps.Lat).rjust(6) + ","\
                  + "Lng:"+str(self.gps.Lon).rjust(6) + ","\
                  + "ax:"+str(self.bno055.ax).rjust(6) + ","\
                  + "ay:"+str(self.bno055.ay).rjust(6) + ","\
                  + "az:"+str(self.bno055.az).rjust(6) + ","\
                  + "rV:"+str(round(self.rightMotor.velocity,3)).rjust(6) + ","\
                  + "lV:"+str(round(self.leftMotor.velocity,3)).rjust(6) + ","\
                  + "q:"+str(self.bno055.ex).rjust(6) + ","\
                  + "Camera:" + str(self.camerastate)
        
        with open('results/control_result.txt',"a")  as test: # [mode] x:ファイルの新規作成、r:ファイルの読み込み、w:ファイルへの書き込み、a:ファイルへの追記
            test.write(datalog + '\n')

    def sequence(self):
        if self.state == 0:#センサ系の準備を行う段階
            self.preparing()
        elif self.state == 1:#放出・降下を行う段階
            self.flying()
        elif self.state == 2:#着陸判定、パラ分離（焼き切り）
            self.dropping()
        elif self.state == 3:#パラシュートから離れる。カメラでの撮影行う
            self.landing()
        elif self.state == 4:#スパースモデリング第一段階
            self.spm_first(True)
        # elif self.state == 5:#スパースモデリング第二段階
        #     self.spm_second()
        # elif self.state == 6:#経路計画段階
        #     self.running()
        # elif self.state == 7:
        #     self.re_learning()
        # elif self.state == 8:#終了
        #     self.finish()
        else:
            self.state = self.laststate #どこにも引っかからない場合何かがおかしいのでlaststateに戻してあげる

    def setup(self):
        self.gps.setupGps()
        # os.system("sudo insmod LoRa_SOFT/soft_uart.ko")
        self.bno055.setupBno()
#         self.radio.setupRadio()
        if self.bno055.begin() is not True:
            print("Error initializing device")
            exit()    

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
#                 self.switchRadio()

    def preparing(self):#時間が立ったら移行
        if self.preparingTime == 0:
            self.preparingTime = time.time()#時刻を取得
            self.RED_LED.led_on()
            self.BLUE_LED.led_off()
            self.GREEN_LED.led_off()

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
    
    def flying(self):#フライトピンが外れる➡︎ボイド缶から放出されたことを検出するステート
        if self.flyingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.flyingTime = time.time()
            self.RED_LED.led_off()
            self.BLUE_LED.led_off()
            self.GREEN_LED.led_off()

        if GPIO.input(ct.const.FLIGHTPIN_PIN) == GPIO.HIGH:#highかどうか＝フライトピンが外れているかチェック
            self.countFlyLoop+=1
            if self.countFlyLoop > ct.const.FLYING_FLIGHTPIN_COUNT_THRE:#一定時間HIGHだったらステート移行
                self.state = 2
                self.laststate = 2       
        else:
            self.countFlyLoop = 0 #何故かLOWだったときカウントをリセット
    
    def dropping(self): #着地判定ステート
        if self.droppingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.droppingTime = time.time()
            self.RED_LED.led_off()
            self.BLUE_LED.led_on()
            self.GREEN_LED.led_off()
      
        #加速度が小さくなったら着地判定
        if (pow(self.bno055.ax,2) + pow(self.bno055.ay,2) + pow(self.bno055.az,2)) < pow(ct.const.DROPPING_ACC_THRE,2):#加速度が閾値以下で着地判定
            self.countDropLoop+=1            
            if self.countDropLoop > ct.const.DROPPING_ACC_COUNT_THRE:#着地判定が複数回行われたらステート以降
                self.state = 3
                self.laststate = 3
        else:
            self.countDropLoop = 0 #初期化の必要あり

    def landing(self):
        if self.landingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.landingTime = time.time()
            self.RED_LED.led_off()
            self.BLUE_LED.led_off()
            self.GREEN_LED.led_on()
            
        if not self.landingTime == 0:
            #焼き切りによるパラ分離
            if self.landstate == 0:
                GPIO.output(ct.const.SEPARATION_PIN,1) #電圧をHIGHにして焼き切りを行う
                if time.time()-self.landingTime > ct.const.SEPARATION_TIME_THRE:
                    GPIO.output(ct.const.SEPARATION_PIN,0) #焼き切りが危ないのでlowにしておく
                    self.landstate = 1
                    self.pre_motorTime = time.time()
        
            #焼き切り終了後カメラ起動
            elif self.landstate == 1:
                self.cap = cv2.VideoCapture(0)
                self.landstate = 2
            
            #カメラ起動後分離シート離脱
            elif self.landstate == 2:
                self.rightMotor.go(ct.const.LANDING_MOTOR_VREF)
                self.leftMotor.go(ct.const.LANDING_MOTOR_VREF)

                if time.time()-self.pre_motorTime > ct.const.LANDING_PRE_MOTOR_TIME_THRE: #5秒間モータ回して分離シートから十分離れる
                    self.rightMotor.stop()
                    self.leftMotor.stop()
                    self.state = 4
                    self.laststate = 4

    def spm_first(self,learn_state):
        #学習用画像を一枚撮影
        if self.camerafirst == 0:
            ret, self.firstlearnimg = self.cap.read()
            cv2.imwrite(f"results/camera_result/first/firstimg{self.firstlearnimgcount}.jpg",self.firstlearnimg)
            self.camerastate = "captured!"
            self.camerafirst = 1
        else:
            self.camerastate = 0

        #フォルダ作成部分
        if not os.path.exists(self.self.saveDir):
            os.mkdir(self.saveDir)
        if not os.path.exists(self.saveDir + f"/bbba_learnimg"):
            os.mkdir(self.saveDir + f"/bbba_learnimg")
        if not os.path.exists(self.saveDir + f"/bcca_secondinput"):
            os.mkdir(self.saveDir + f"/bcca_secondinput")
        saveName = self.saveDir + f"/bcba_difference"
        if not os.path.exists(saveName):
            os.mkdir(saveName)

        start_time = time()#学習用時間計測。学習開始時間
        
        #保存時のファイル名指定（現在は時間）
        now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
        saveName = self.saveDir + f"/bcba_difference/{now}"
        if not os.path.exists(saveName):
            os.mkdir(saveName)
        Save = True
        
        # Path that img will be read
        #importPath = path.replace("\\", "/")
        importPath = self.firstlearnimg
        
        # This will change such as datetime
        # print("CURRENT FRAME: "+str(re.findall(".*/frame_(.*).jpg", importPath)[0]))
        
        iw_shape = (2, 3)#ウィンドウのシェイプ
        D, ksvd = None, None #最初に指定しないと怒られちゃうから
        feature_values = {}

        if learn_state:
            print("=====LEARNING PHASE=====")
        else:
            print("=====EVALUATING PHASE=====")
            
        iw = IntoWindow(importPath, self.saveDir, Save) #画像の特徴抽出のインスタンス生成
        # processing img
        fmg_list = iw.feature_img(frame_num=now) #特徴抽出。リストに特徴画像が入る
        
        if learn_state:#学習モデル獲得
            for fmg in fmg_list:#それぞれの特徴画像に対して処理
                # breakout by windows
                iw_list, window_size = iw.breakout(iw.read_img(fmg)) #ブレイクアウト
                feature_name = str(re.findall(self.saveDir + f"/baca_featuring/(.*)_.*_", fmg)[0])
                print("FEATURED BY: ",feature_name)

                for win in range(int(prod(iw_shape))): #それぞれのウィンドウに対して学習を実施
                    if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                        ld = LearnDict(iw_list[win])
                        D, ksvd = ld.generate() #辞書獲得
                        self.dict_list[feature_name] = [D, ksvd]
                        save_name = self.saveDir + f"/bbba_learnimg/{feature_name}_part_{win+1}_{now}.jpg"
                        cv2.imwrite(save_name, iw_list[win])
            learn_state = False

        else:#20枚撮影
            for i in range(ct.const.SPM_FIRST_COUNT_THRE):
                for fmg in fmg_list:#それぞれの特徴画像に対して処理
                    iw_list, window_size = iw.breakout(iw.read_img(fmg)) #ブレイクアウト
                    feature_name = str(re.findall(self.saveDir + f"/baca_featuring/(.*)_.*_", fmg)[0])
                    print("FEATURED BY: ",feature_name)
                    
                    for win in range(int(prod(iw_shape))): #それぞれのウィンドウに対して学習を実施
                        D, ksvd = self.dict_list[feature_name]
                        ei = EvaluateImg(iw_list[win])
                        img_rec = ei.reconstruct(D, ksvd, window_size)
                        saveName = self.saveDir + f"/bcba_difference"
                        if not os.path.exists(saveName):
                            os.mkdir(saveName)
                        saveName = self.saveDir + f"/bcba_difference/{now}"
                        if not os.path.exists(saveName):
                            os.mkdir(saveName)
                        ave, med, var, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, self.saveDir)
                        #if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                        #    feature_values[feature_name] = {}
                        #    feature_values[feature_name]["var"] = ave
                        #    feature_values[feature_name]["med"] = med
                        #    feature_values[feature_name]["ave"] = var

                        if win == 0:
                            feature_values[feature_name] = {}

                        feature_values[feature_name][f'win_{win+1}'] = {}
                        feature_values[feature_name][f'win_{win+1}']["var"] = ave
                        feature_values[feature_name][f'win_{win+1}']["med"] = med
                        feature_values[feature_name][f'win_{win+1}']["ave"] = var
                        # feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
                        # feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
                        
                self.rightMotor.go(ct.const.SPM_MOTOR_VREF)#走行
                self.leftMotor.go(ct.const.SPM_MOTOR_VREF)#走行
                time.sleep(2)
                self.rightMotor.stop()
                self.leftMotor.stop()

        # for fmg in fmg_list:#それぞれの特徴画像に対して処理
        #     # breakout by windows
        #     iw_list, window_size = iw.breakout(iw.read_img(fmg)) #ブレイクアウト
        #     feature_name = str(re.findall(self.saveDir + f"/baca_featuring/(.*)_.*_", fmg)[0])
        #     print("FEATURED BY: ",feature_name)

        #     if learn_state: #1枚撮影して学習モデル獲得
        #         for win in range(int(prod(iw_shape))): #それぞれのウィンドウに対して学習を実施
        #             if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
        #                 ld = LearnDict(iw_list[win])
        #                 D, ksvd = ld.generate() #辞書獲得
        #                 self.dict_list[feature_name] = [D, ksvd]
        #                 save_name = self.saveDir + f"/bbba_learnimg/{feature_name}_part_{win+1}_{now}.jpg"
        #                 cv2.imwrite(save_name, iw_list[win])


        #     else:
        #         for i in range(ct.const.SPM_FIRST_COUNT_THRE):
        #             self.rightMotor.go(ct.const.SPM_MOTOR_VREF)#走行
        #             self.leftMotor.go(ct.const.SPM_MOTOR_VREF)#走行
        #             D, ksvd = self.dict_list[feature_name]
        #             ei = EvaluateImg(iw_list[win])
        #             img_rec = ei.reconstruct(D, ksvd, window_size)
        #             saveName = self.saveDir + f"/bcba_difference"
        #             if not os.path.exists(saveName):
        #                 os.mkdir(saveName)
        #             saveName = self.saveDir + f"/bcba_difference/{now}"
        #             if not os.path.exists(saveName):
        #                 os.mkdir(saveName)
        #             ave, med, var, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, self.saveDir)
        #             #if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
        #             #    feature_values[feature_name] = {}
        #             #    feature_values[feature_name]["var"] = ave
        #             #    feature_values[feature_name]["med"] = med
        #             #    feature_values[feature_name]["ave"] = var
                    
        #             if win == 0:
        #                 feature_values[feature_name] = {}

        #             feature_values[feature_name][f'win_{win+1}'] = {}
        #             feature_values[feature_name][f'win_{win+1}']["var"] = ave
        #             feature_values[feature_name][f'win_{win+1}']["med"] = med
        #             feature_values[feature_name][f'win_{win+1}']["ave"] = var
        #             # feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
        #             # feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
        # learn_state = False
                    
        if not learn_state:#npzファイル形式で計算結果保存
            print(feature_values)
            np.savez_compressed(self.saveDir + f"/results/camera_result/processed/secondinput/"+now,array_1=np.array([feature_values]))
        
        end_time = time()#計算終了
        print("Calc Time:",end_time-start_time)
        # Learn state should be changed by main.py
        learn_state = False#学習終了

    def second_spm(self):
        return 0

    def sendRadio(self):
        datalog = str(self.state) + ","\
                  + str(self.gps.Time) + ","\
                  + str(self.gps.Lat) + ","\
                  + str(self.gps.Lon) + ","\

        self.radio.sendData(datalog) #データを送信
        
    def switchRadio(self):
        datalog = str(self.state) + ","\
                  + str(self.gps.Time) + ","\
                  + str(self.gps.Lat) + ","\
                  + str(self.gps.Lon) + ","\

        self.radio.switchData(datalog) #データを送信
        
    def run_motor(self):
        self.rightMotor.go(ct.const.MOTOR_VREF)
        self.leftMotor.go(ct.const.MOTOR_VREF)
    
    def camera(self,cap):
        ret, img = cap.read()      
        cv2.imshow('camera', img)
        return img
    
    def stuck_detection(self):
        return 0

    def keyboardinterrupt(self):
        self.rightMotor.stop()
        self.leftMotor.stop()
        self.RED_LED.led_off()
        self.BLUE_LED.led_off()
        self.GREEN_LED.led_off()
        self.cap.release()
        cv2.destroyAllWindows()
