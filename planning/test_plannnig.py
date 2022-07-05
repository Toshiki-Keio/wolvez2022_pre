# ラズパイで動かす時にはこれコメントアウトをはずこと
# import motor
# import estimation
# import constant as ct
# import RPi.GPIO as GPIO
# import time
# import bno055

import time
import math
from math import sqrt
from math import radians
from math import sin
from math import fabs
import numpy as np


# GPSの値を取得後，指定したものと比較してその方向に走行する
# 昨年のコードを参照したい！
# http://hamasyou.com/blog/2010/09/07/post-2/

# 現在の緯度と経度を取得

# ゴール地点の緯度と経度を指定

# 2点の情報から方向・距離を算出






# 危険度を仮設定
# あとで林出にデータをもらう！
# degree_of_risk = np.zeros((2,3))
degree_of_risk = np.asarray([[40,40,30],[40,40,40]])

print(degree_of_risk)

# 危険度の最小値のインデックスを取得
min_idx = np.unravel_index(np.argmin(degree_of_risk), degree_of_risk.shape)
print(min_idx[0])

# 選ばれた領域に対する動きを設定
# モータの設定
""" 
GPIO.setwarnings(False)
Motor1 = motor2.motor(6,5,13)    # おそらく右のモータ
Motor2 = motor2.motor(20,16,12)  # おそらく左のモータ
"""
# 0列目（左側）が選ばれた場合には左回転
if min(min_idx[1]) == 0:
    """ 
    Motor1.go(70)
    Motor2.back(70)
    time.sleep(5) 
    """
    # 再度危険度を算出

# 1列目（中央）が選ばれた場合には直進
elif min(min_idx[1]) == 1:
    """ 
    Motor1.go(70)
    Motor2.back(70)
    time.sleep()
    """
# 2列目（右側）が選ばれた場合には右回転
elif min(min_idx[1]) == 2:
    """ 
    Motor1.back(70)
    Motor2.go(70)
    time.sleep
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




