from getkey import getkey
from multiprocessing import Pipe, Process
import time


def keyboard_event(child):
    st = time.time()
    while True: #Breaks when key is pressed
        key = getkey()
        ct = time.time()
        if key == "t" and ct - st > 1.0:
            st = time.time()
            child.send([1])
