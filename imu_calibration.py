import numpy as np
import imu_core.imu_core_cpp as IMU

import meshcat
import pinocchio as pin
import time
from scipy.spatial.transform import Rotation
from imu_measurements import ArmMeasurement

# imu_base = IMU.Imu3DM_GX3_45("/dev/ttyACM0", True)
# imu_base.initialize()

# imu_shoulder = IMU.Imu3DM_GX3_45("/dev/ttyACM1", True)
# imu_shoulder.initialize()

# imu_palm = IMU.Imu3DM_GX3_45("/dev/ttyACM2", True)
# imu_palm.initialize()

imus = ArmMeasurement("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2")

X_TG = np.eye(4)
X_TG[0,3] = 0.05
Y_TG = np.eye(4)
Y_TG[1,3] = 0.05
Z_TG = np.eye(4)
Z_TG[2,3] = 0.05

vis = meshcat.Visualizer().open()
xbox = meshcat.geometry.Box([0.1, 0.01, 0.01])
vis["xbox1"].set_object(xbox, meshcat.geometry.MeshLambertMaterial(
                                 color=0xFF0000))
ybox = meshcat.geometry.Box([0.01, 0.1, 0.01])
vis["ybox1"].set_object(ybox, meshcat.geometry.MeshLambertMaterial(
                                 color=0x00FF00))
zbox = meshcat.geometry.Box([0.01, 0.01, 0.1])
vis["zbox1"].set_object(zbox, meshcat.geometry.MeshLambertMaterial(
                                 color=0x0000FF))

orientation = np.zeros(4)

counter = 0
# bRw = pin.utils.rpyToMatrix(0, 0, np.pi/2.0)  @ pin.utils.rpyToMatrix(np.pi/2.0, 0, 0) 
# wRb = bRw.T
T = np.eye(4)
# wRa = pin.Quaternion(pin.utils.rpyToMatrix(np.pi/2.0,0,0) @ pin.utils.rpyToMatrix(0,-np.pi/2.0,0))
while True:
    # iRb = (imu_base.get_rotation_matrix()).copy().T
    # iRb[:,0] *= -1
    # iRb[:,2] *= -1
    # iRs = (pin.Quaternion(imu_shoulder.get_quaternion_wxyz()).toRotationMatrix()).T
    # iRs = (imu_shoulder.get_rotation_matrix()).copy().T
    # iRs[:,0] *= -1
    # iRs[:,2] *= -1
     
    # iRp = pin.Quaternion(imu_palm.get_quaternion_wxyz()).toRotationMatrix()
    # print(acel)
    imus.update_measurements()

    T[0:3,0:3] = imus.arm_orientation
    vis["xbox1"].set_transform(T @ X_TG)
    vis["ybox1"].set_transform(T @ Y_TG)
    vis["zbox1"].set_transform(T @ Z_TG)
    # print(T[0:3,0:3])

    # T[0:3,0:3] = iRs.toRotationMatrix()
    # vis["xbox2"].set_transform(T)
    # vis["ybox2"].set_transform(T)
    # vis["zbox2"].set_transform(T)

    counter += 1
    time.sleep(0.01)
