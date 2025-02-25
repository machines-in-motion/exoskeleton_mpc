import numpy as np
import collections

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer as Visualizer
import time

from arm_model import create_arm

from estimation_problem import solve_estimation_problem
from interface import ExoSkeletonUDPInterface
from utils import ArmMeasurementCurrent

interface = ExoSkeletonUDPInterface()
counter = 0
interface.calibrate()

rmodel, rdata, gmodel, cmodel = create_arm()

viz = Visualizer(rmodel, gmodel, cmodel)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True

estimate_x0 = np.zeros(rmodel.nq + rmodel.nv)

T = 30
measurement = collections.deque(maxlen=T)
counter = 0

print("reading measurements")

while True:
    interface.setCommand([0], [0.], [0], [0], [0.0])
    time.sleep(0.01)
    state = interface.getState()
    try:
        measurement.append(ArmMeasurementCurrent(state))
        if counter > T:
            estimate = solve_estimation_problem(measurement, T, rmodel, estimate_x0)
            viz.display(estimate.xs[-1][:rmodel.nq])
            estimate_x0 = estimate.xs[-1]
        print(estimate_x0[1]- state["q"], state["motor_q"])
        # time.sleep(0.01)
        counter += 1
    except:
        print("oops")
del imus