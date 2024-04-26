import numpy as np
from interface import ExoSkeletonUDPInterface

import meshcat
import pinocchio as pin
import time
from scipy.spatial.transform import Rotation

from utils import add_frame, update_frame

interface = ExoSkeletonUDPInterface()
counter = 0
interface.calibrate()

vis = meshcat.Visualizer().open()
add_frame("arm", vis)
add_frame("wrist", vis)
add_frame("diff", vis)

# add_frame("wrist", vis)

tmp1 = pin.utils.rpyToMatrix(0, 0.0, 0)
tmp2 = pin.utils.rpyToMatrix(0,  -np.pi/2.0,0) #@ pin.utils.rpyToMatrix(-np.pi/2.0, 0, 0)

counter = 0
while True:
    interface.setCommand([0], [0.], [0], [0], [0.0])
    time.sleep(0.001)

    state = interface.getState()

    base = Rotation.from_quat(np.squeeze(state["base_ori"])).as_matrix()
    shoulder = Rotation.from_quat(np.squeeze(state["shoulder_ori"])).as_matrix()
    # wrist = Rotation.from_quat(np.squeeze(state["wrist_ori"])).as_matrix()
    update_frame("wrist", vis, shoulder)
    update_frame("arm", vis, base, [0.2, 0, 0])
    update_frame("diff", vis, tmp2@base.T@shoulder, [0.0, 0.2, 0])
    counter += 1
    print("issue", counter)

    