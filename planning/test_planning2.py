# ラズパイで動かす時にはこれコメントアウトをはずこと
# import motor
# import estimation
# import constant as ct
# import RPi.GPIO as GPIO
# import bno055
# import gps

from msilib import type_string
import time
import math
from math import sqrt
from math import radians
from math import sin
from math import fabs
from turtle import distance
import numpy as np

# ゴール方向の角度から行く方向を決定する関数
def decide_direction(direction_goal_deg):
    if direction_goal_deg >= 20:
        direction_goal_lcr = 2
        print("ゴール方向："+str(direction_goal_lcr)+" 右に曲がりたい")
    elif direction_goal_deg > -20 and direction_goal_deg < 20:
        direction_goal_lcr = 1
        print("ゴール方向："+str(direction_goal_lcr)+" 直進したい")
    else:
        direction_goal_lcr = 0
        print("ゴール方向："+str(direction_goal_lcr)+" 左に曲がりたい")
    return direction_goal_lcr

# それぞれの方向に対して実際に行う動作を決める関数
def decide_behavior(direction):
    if direction == 0:
        print("左に曲がる")
    elif direction == 1:
        print("直進する")
    elif direction == 2:
        print("右に曲がる")

# GPSから距離・方向を算出する関数
# 緯度：latitude，経度：longitude
def direction_from_gps(now_lat,now_lon,goal_lat,goal_lon):


    distance_goal = 10
    direction_goal_deg = 10
    return [direction_goal_deg, distance_goal]

# GPS起動
# gps = gps.GPS()
# gps.setup()

count = 0
while count < 5:
    count += 1
    # 危険度行列
    # risk : spm2から出力された危険度
    risk = np.random.randint(0,100,(2,3))
    upper_risk = risk[0,:]
    lower_risk = risk[1,:]
    print("risk:\n"+str(risk)+"\n")
    threshold_risk = 70


    if np.amin(lower_risk) >= threshold_risk:
        print("前方に安全なルートはありません。90度回転して新たな経路を探索します。")
    else:
        # ゴール方向の算出
        direction_goal_deg = np.random.randint(-60,60)  #ゴール方向の角度を取得（後でGPSから値が取れるようにする）
        direction_goal_lcr = decide_direction(direction_goal_deg)  #角度から左・前・右のどの方向に進むべきかを取得
        direction_real = direction_goal_lcr

        if lower_risk[direction_goal_lcr] <= threshold_risk:   #ゴール方向の危険度が閾値以下の場合
            decide_behavior(direction_real)   # その方向に進む
        else:
            print("ゴール方向が安全ではありません。別ルートを探索します。")
            if direction_goal_lcr == 0:
                if lower_risk[1] <= lower_risk[2]:
                    direction_real = 1
                else:
                    direction_real = 2
                decide_behavior(direction_real)
            elif direction_goal_lcr == 1:
                if lower_risk[0] <= lower_risk[2]:
                    direction_real = 0
                else:
                    direction_real = 2
                direction_real = 0
                decide_behavior(direction_real)
            elif direction_goal_lcr == 2:
                if lower_risk[0] <= lower_risk[1]:
                    direction_real = 0
                else:
                    direction_real = 1
                decide_behavior(direction_real)


