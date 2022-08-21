import RPi.GPIO as gpio
import time

class RaspPinout():
    def __init__(self):
        gpio.setmode(gpio.BOARD)
        # gpio.setmode(gpio.BCM)
    
    def control(self, auth):
        if auth is True:

            gpio.setup(5, gpio.OUT)
            gpio.output(5, gpio.HIGH)
            gpio.output(7, gpio.LOW)
            time.sleep(15)
            gpio.output(5, gpio.LOW)
