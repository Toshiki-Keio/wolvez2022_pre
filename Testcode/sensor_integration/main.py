import cv2
#import gps
#from MicropyGPS import MicropyGPS
#from LoRa_SOFT.LoRa import LoRa
import sys
sys.path.append("/home/pi/Desktop/wolvez2021/Testcode/sensor_integration/LoRa_SOFT")
from bno055 import BNO055
from encoder_motor2 import motor
from servomotor import servomotor
import time
import RPi.GPIO as GPIO
from cansat import Cansat

cansat = Cansat()
cansat.setup()

try:
    while True:
        cansat.run()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
except KeyboardInterrupt:
    print("Finished")
    cansat.keyboardinterrupt()
    GPIO.cleanup()
