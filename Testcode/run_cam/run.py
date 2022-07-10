import RPi.GPIO as GPIO
import time
import constant as ct
import cv2
from motor import motor
from bno055 import BNO055

try:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM) #GPIOの設定

    
    #Motor
    rightMotor = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
    leftMotor = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)

    #camera
    cap = cv2.VideoCapture(0)

    while True:
        #モータ回転
        rightMotor.go(ct.const.MOTOR_VREF)
        leftMotor.go(ct.const.MOTOR_VREF)

        datalog = "rV:" + str(round(rightMotor.velocity,2)).rjust(6) + ","\
                        + "lV:" + str(round(leftMotor.velocity,2)).rjust(6)
        print(datalog)
#         time.sleep(0.3)
    
except KeyboardInterrupt:
    rightMotor.stop()
    leftMotor.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()