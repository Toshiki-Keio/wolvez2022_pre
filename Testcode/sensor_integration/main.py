import cv2
import RPi.GPIO as GPIO
import Cansat

cansat = Cansat()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM) #GPIOの設定
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