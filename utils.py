import meshcat
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

# This class takes the current imu orienation and computes the arm and shoulder orinetation in the world frame
@dataclass
class ArmMeasurementCurrent:
    arm_orientation : np.array
    palm_orientation : np.array
    imu_to_world = pin.utils.rpyToMatrix(0, -np.pi/2.0, 0)

    def __init__(self, state, wrist_orientation):
        base = Rotation.from_quat(np.squeeze(state["base_ori"])).as_matrix()
        shoulder = Rotation.from_quat(np.squeeze(state["shoulder_ori"])).as_matrix()
        wrist = Rotation.from_quat(wrist_orientation).as_matrix()
        self.arm_orientation = self.imu_to_world@base.T@shoulder
        self.palm_orientation = self.imu_to_world@base.T@wrist

def add_frame(name, vis):
    xbox = meshcat.geometry.Box([0.1, 0.01, 0.01])
    vis["xbox_" + name].set_object(xbox, meshcat.geometry.MeshLambertMaterial(
                                    color=0xFF0000))
    ybox = meshcat.geometry.Box([0.01, 0.1, 0.01])
    vis["ybox_" + name].set_object(ybox, meshcat.geometry.MeshLambertMaterial(
                                    color=0x00FF00))
    zbox = meshcat.geometry.Box([0.01, 0.01, 0.1])
    vis["zbox_" + name].set_object(zbox, meshcat.geometry.MeshLambertMaterial(
                                    color=0x0000FF))

def update_frame(name, vis, R, offset = np.zeros(3)):
    X_TG = np.eye(4)
    X_TG[0,3] = 0.05
    Y_TG = np.eye(4)
    Y_TG[1,3] = 0.05
    Z_TG = np.eye(4)
    Z_TG[2,3] = 0.05

    offset_TG = np.eye(4)
    offset_TG[0:3,3] = offset


    T = np.eye(4)
    T[0:3,0:3] = R
    vis["xbox_" + name].set_transform( offset_TG @ T @ X_TG )
    vis["ybox_" + name].set_transform( offset_TG @ T @ Y_TG )
    vis["zbox_" + name].set_transform( offset_TG @ T @ Z_TG )


## This is the python interface to read IMU data from the BNU005 IMU. 

import serial
from threading import Thread
import struct

class BnoImu:
    def __init__(self, port = '/dev/ttyACM0'):
        self.port = port
        self.serial = serial.Serial(port, baudrate=115200, timeout=1)
        self.running = True
        self.state = None
        self.thread = Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            data = self.serial.read_until(b'abc\n')
            data = struct.unpack('16f',data[:-4])
            self.state = {'q':data[0:4],
                          'acce':data[4:7],
                          'gyro':data[7:10],
                          'mag':data[10:13],
                          'gravity':data[13:16]}
            
    def read(self):
        return self.state
    
    def close(self):
        self.running=False
        self.serial.close()