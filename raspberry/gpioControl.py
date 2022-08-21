import time
import RPi.GPIO as gpio # GPIO Library

class RaspPinout():
    def __init__(self):
        gpio.setmode(gpio.BOARD)
        # gpio.setmode(gpio.BCM)
    
    def control(self, auth):
        # Setup GPIO
        gpio.setup(5, gpio.OUT) # Green LED
        gpio.setup(7, gpio.OUT) # Red LED

        if auth is True: # Authentication successful

            gpio.output(5, gpio.HIGH) # Green LED high
            gpio.output(7, gpio.LOW) # Red LED low
            time.sleep(5) # wait 5 seconds
            gpio.output(5, gpio.LOW) # Turn off green LED
        else:
            # Both LEDs turning off
            gpio.output(5, gpio.LOW)
            gpio.output(7, gpio.HIGH)
            