import serial
from threading import Thread
import struct
import numpy as np

class ExoIMUs:
    def __init__(self, port = '/dev/ttyACM1'):
        self.port = port
        self.serial = serial.Serial(port, baudrate=115200, timeout=1)
        self.running = True
        self.state = None
        self.thread = Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            data = self.serial.read_until(b'abc\n')
            if(len(data) == 4 + 3*(4*4 + 4)):
                data = struct.unpack('15f',data[:-4])
                self.state = {'q_hand':np.asarray(data[0:4]),
                            'q_shoulder':np.asarray(data[4:8]),
                            'q_base':np.asarray(data[8:12]),
                            'hand_acc': data[12],
                            'shoulder_acc': data[13],
                            'base_acc':data[14]}
            
    def read(self):
        return self.state
    
    def close(self):
        self.running=False
        self.serial.close()