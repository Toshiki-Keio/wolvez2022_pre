#Last Update 2022/07/02
#Author : Toshiki Fukui

import RPi.GPIO as GPIO
from cansat import Cansat
import time

state = 0

cansat = Cansat(state)
cansat.setup()

try:
    while True:
        cansat.sensor()
        time.sleep(0.3)
        cansat.sequence()
        if cansat.state >= 5:
            print("Finished")
            cansat.keyboardinterrupt()
            GPIO.cleanup()
            break
    
except KeyboardInterrupt:
    print("Finished")
    cansat.keyboardinterrupt()
    GPIO.cleanup()