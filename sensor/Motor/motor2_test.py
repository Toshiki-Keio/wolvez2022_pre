import motor2
import RPi.GPIO as GPIO
import time 

GPIO.setwarnings(False)
Motor1 = motor2.motor(6,5,13)
Motor2 = motor2.motor(20,21,12)

try:
    print("motor run") 
    Motor1.go(60)
    Motor2.go(60)
    time.sleep(5)

    #Motor.back(100)
    #time.sleep(3)
    print("motor stop")
    Motor1.stop()
    Motor2.stop()
    time.sleep(1)
except KeyboardInterrupt:
    Motor1.stop()
    Motor2.stop()
    GPIO.cleanup()

GPIO.cleanup()

