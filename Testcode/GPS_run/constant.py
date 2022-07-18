#Last Update 2022/07/02
#Author : Toshiki Fukui

import const

##Pin Number
#Motor&Encoder
const.RIGHT_MOTOR_IN1_PIN = 6
const.RIGHT_MOTOR_IN2_PIN = 5
const.RIGHT_MOTOR_VREF_PIN = 13
const.RIGHT_MOTOR_ENCODER_A_PIN = 26
const.RIGHT_MOTOR_ENCODER_B_PIN = 19

const.LEFT_MOTOR_IN1_PIN = 20
const.LEFT_MOTOR_IN2_PIN = 16
const.LEFT_MOTOR_VREF_PIN = 12
const.LEFT_MOTOR_ENCODER_A_PIN = 7
const.LEFT_MOTOR_ENCODER_B_PIN = 8

#LED
const.RED_LED_PIN =  10
const.BLUE_LED_PIN = 9
const.GREEN_LED_PIN = 11

#Releasing Pin
const.SEPARATION_PIN = 25

#Flight Pin
const.FLIGHTPIN_PIN = 4

#Motor
const.LANDING_MOTOR_VREF = 90
const.SPM_MOTOR_VREF = 70
const.RUNNING_MOTOR_VREF = 70
const.STUCK_MOTOR_VREF = 100



##Variable Threshold
const.ANGLE_THRE = 10
const.SHADOW_EDGE_LENGTH = 15
const.CASE_DISCRIMINATION = 1 #Case判定における許容誤差
const.START_CONST_SHORT = 0.5 #Startingステートにおける帯の幅　±0.5
const.START_CONST_LONG = 5 #Startingステートにおける帯の幅　±5


##State Threshold
const.PREPARING_GPS_COUNT_THRE= 30
const.PREPARING_TIME_THRE = 10
const.FLYING_FLIGHTPIN_COUNT_THRE = 10
const.DROPPING_ACC_COUNT_THRE = 20
const.DROPPING_ACC_THRE = 1 #加速度の値
const.SEPARATION_TIME_THRE = 10 #焼き切り時間
const.LANDING_CAMERA_TIME_THRE = 2

const.LANDING_MOTOR_TIME_THRE = 5 #分離シートから離れるためにモータを回転させる時間
const.STUCK_ACC_THRE = 0.1
const.STUCK_COUNT_THRE = 10
const.STUCK_MOTOR_TIME_THRE = 5 #分離シートから離れるためにモータを回転させる時間
const.LANDING_PRE_MOTOR_TIME_THRE = 5 #分離シートから離れるためにモータを回転させる時間
const.SPMFIRST_PIC_COUNT = 50
const.STUCK_START = 11
const.STUCK_END = 13
const.STARTING_TIME_THRE = 60
const.MEASURING_SWITCH_COUNT_THRE = 20 #1地点での測位回数
const.MEASURING_MAX_MEASURING_COUNT_THRE = 6 #最大測位点