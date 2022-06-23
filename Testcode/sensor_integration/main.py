import cv2
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