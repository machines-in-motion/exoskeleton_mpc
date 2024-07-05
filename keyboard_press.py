from getkey import getkey
from multiprocessing import Pipe, Process


def keyboard_event(child):
    while True: #Breaks when key is pressed
        key = getkey()
        if key == "t":
            child.send([1])
