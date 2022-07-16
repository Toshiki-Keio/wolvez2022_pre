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
import numpy as np

# 危険度行列
# risk : spm2から出力された危険度
# bias_goal : Cansatとゴール方向の相対角度から算出されるバイアス行列
# integrated_risk : 上記2つの行列を足し合わせたもの


# GPSの値を取得後，指定したものと比較してその方向に走行する
# 昨年のコードを参照したい！
# http://hamasyou.com/blog/2010/09/07/post-2/

# 現在の緯度と経度を取得

# ゴール地点の緯度と経度を指定

# 2点の情報から方向・距離を算出

direction_cansat2goal = 0*180/math.pi # 仮にCansatとゴールの相対角度を指定
bias_goal = np.zeros((2,3)) # ゴール方向に対するバイアスの初期設定
num_bias = -10
# Cansatとゴールの相対角度からバイアス行列を設定
if direction_cansat2goal >= 20:
    np.put(bias_goal,[2,5],num_bias)
elif direction_cansat2goal > -20 and direction_cansat2goal < 20:
    np.put(bias_goal,[1,4],num_bias)
else:
    np.put(bias_goal,[0,3],num_bias)
print("bias_goal: \n"+str(bias_goal)+"\n") #確認用



# 危険度を仮設定
# あとで林出にデータをもらう！
# degree_of_risk = np.zeros((2,3))
# risk = np.asarray([[40,40,30],[40,40,40]])
risk = np.random.randint(30,100,(2,3))
upper_risk = risk[0,:]
lower_risk = risk[1,:]
print("risk:\n"+str(risk)+"\n")
# print("upper_risk:\n"+str(upper_risk)+"\n")
# print("lower_risk:\n"+str(lower_risk)+"\n")

# バイアスをかけた危険度
integrated_risk = risk + bias_goal
upper_integrated_risk = integrated_risk[0,:]
lower_integrated_risk = integrated_risk[1,:]
print("integrated_risk: \n" + str(integrated_risk)+"\n")
# print("upper_integrated_risk: \n" + str(upper_integrated_risk)+"\n")
# print("lower_integrated_risk: \n" + str(lower_integrated_risk)+"\n")

# 危険度の最小値のインデックスを取得
lower_risk_min_idx = np.argmin(lower_risk)
print("lower_risk_min_idx: "+str(lower_risk_min_idx))
lower_integrated_risk_min_idx = np.argmin(lower_integrated_risk)
print("lower_integrated_risk_min_idx: "+str(lower_integrated_risk_min_idx))

print("＜もともとの危険度のみから判断した場合＞")
if np.amin(lower_risk) >= 80:
    print("riskの下の行が全て閾値以上")
    print("どこも危険なので旋回して別角度を探索")
else:
    if lower_risk_min_idx == 0:
        print("左側が安全そうなので左に曲がりながら走行")
    elif lower_risk_min_idx == 1:
        print("中央が安全そうなので直進")
    elif lower_risk_min_idx == 2:
        print("右側が安全そうなので右に曲がりながら走行")

print("＜バイアスありの危険度から判断した場合＞")
if np.amin(lower_integrated_risk) >= 80:
    print("riskの下の行が全て閾値以上")
    print("どこも危険なので旋回して別角度を探索")
else:
    if lower_integrated_risk_min_idx == 0:
        print("左側が安全そうなので左に曲がりながら走行")
    elif lower_integrated_risk_min_idx == 1:
        print("中央が安全そうなので直進")
    elif lower_integrated_risk_min_idx == 2:
        print("右側が安全そうなので右に曲がりながら走行")

# 選ばれた領域に対する動きを設定
# モータの設定
""" 
GPIO.setwarnings(False)
Motor1 = motor2.motor(6,5,13)    # おそらく右のモータ
Motor2 = motor2.motor(20,16,12)  # おそらく左のモータ
"""


# 以下，estimation_test.pyから改良
# GPIO.setwarnings(False)
# MotorR = motor.motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
# MotorL = motor.motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN,ct.const.LEFT_MOTOR_VREF_PIN)
# # Encoder = estimation.estimation(ct.const.RIGHT_MOTOR_ENCODER_A_PIN,ct.const.RIGHT_MOTOR_ENCODER_B_PIN,ct.const.LEFT_MOTOR_ENCODER_A_PIN,ct.const.LEFT_MOTOR_ENCODER_B_PIN)
# bno055 = bno055.BNO055()

# bno055.setupBno()
# x=0
# y=0
# q=0
# del_t=0.2
# hantei = 0
# state = 1
# k = 20
# v_ref = 70
# x_remind = []
# y_remind = []
# q_remind = []
# # MotorR.stop()
# # MotorL.stop()

# start_time=time.time()
# print("cansat-x :",x,"[m]")
# print("cansat-y :",y,"[m]")
# print("cansat-q :",q,"[rad]")




