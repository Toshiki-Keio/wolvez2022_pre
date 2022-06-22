import RPi.GPIO as GPIO
import Cansat
import time

state = 0

cansat = Cansat(state)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM) #GPIOの設定
cansat.setup()

try:
    while True:
        cansat.sensor()
        time.sleep(0.05)
        cansat.sequence()
    
except KeyboardInterrupt:
    print("Finished")
    GPIO.cleanup()