## This class reads data from the IMUs
## Author : Avadesh Meduri
## Date : 1/03/2024

import pinocchio as pin
import numpy as np
import imu_core.imu_core_cpp as IMU
from dataclasses import dataclass

@dataclass
class ArmMeasurement:
    arm_orientation : np.array
    palm_orientation : np.array
    
    def __init__(self, port_base_imu, port_shoulder_imu, port_palm_imu):
        """
        base_wrist_imu : port name of the base imu attached to the body
        port_shoulder_imu : port name of the shoulder imu
        port_palm_imu : port name of the palm imu attached to the hand/wrist
        """
        self.imu_base = IMU.Imu3DM_GX3_45(port_base_imu, True)
        self.imu_palm = IMU.Imu3DM_GX3_45(port_palm_imu, True)
        self.imu_shoulder = IMU.Imu3DM_GX3_45(port_shoulder_imu, True)

        self.imu_base.initialize() 
        self.imu_palm.initialize() 
        self.imu_shoulder.initialize()

        ## Rotation from the base frame to the desired world frame
        self.bRw = pin.utils.rpyToMatrix(0, 0, -np.pi/2.0) @ pin.utils.rpyToMatrix(np.pi/2.0, 0, 0) 

    def update_measurements(self):

        self.base_orientation = self.imu_base.get_rotation_matrix().T
        arm_orientation = self.imu_shoulder.get_rotation_matrix().T
        self.arm_orientation =  self.bRw @ self.base_orientation.T @ arm_orientation 
        palm_orientation = self.imu_palm.get_rotation_matrix().T
        self.palm_orientation = self.bRw @ self.base_orientation.T @ palm_orientation


if __name__ == "__main__":
    imus = ArmMeasurement("/dev/ttyACM0", "/dev/ttyACM3")
    for i in range(10000):
        imus.update_measurements()  
        print(imus.arm_orientation, imus.palm_orientation)
    
    del imus