import numpy as np
from interface import ExoSkeletonUDPInterface

import meshcat
import pinocchio as pin
import time
from scipy.spatial.transform import Rotation
from imu_measurements import ArmMeasurement

from utils import add_frame, update_frame, BnoImu

interface = ExoSkeletonUDPInterface()
counter = 0
interface.calibrate()

vis = meshcat.Visualizer().open()
add_frame("arm", vis)
# add_frame("wrist", vis)

wrist_imu = BnoImu("/dev/ttyACM1")
time.sleep(0.01)

tmp1 = pin.utils.rpyToMatrix(0, -np.pi/2.0, 0)
tmp2 = pin.utils.rpyToMatrix(0,  -np.pi/2.0,0) #@ pin.utils.rpyToMatrix(-np.pi/2.0, 0, 0)

while True:
    interface.setCommand([0], [0.], [0], [0], [0.0])
    time.sleep(0.005)

    state = interface.getState()

    base = Rotation.from_quat(np.squeeze(state["base_ori"])).as_matrix()
    shoulder = Rotation.from_quat(np.squeeze(state["shoulder_ori"])).as_matrix()
    wrist = Rotation.from_quat(np.squeeze(wrist_imu.read()["q"])).as_matrix()

    update_frame("arm", vis, tmp1@base.T@shoulder)
    # update_frame("wrist", vis, tmp@base.T@wrist)