import numpy as np
import collections
from dataclasses import dataclass

import pinocchio as pin
import hppfcl as fcl
from pinocchio.visualize import MeshcatVisualizer as Visualizer
import time
from matplotlib import pyplot as plt
import meshcat

import crocoddyl
from mim_solvers import SolverSQP

from reaching_problem import solve_reaching_problem
from arm_model import create_arm

from imu_measurements import ArmMeasurement
from estimation_problem import solve_estimation_problem


rmodel, rdata, gmodel, cmodel = create_arm()

viz = Visualizer(rmodel, gmodel, cmodel)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True


X_TG = np.eye(4)
X_TG[0,3] = 0.05
Y_TG = np.eye(4)
Y_TG[1,3] = 0.05
Z_TG = np.eye(4)
Z_TG[2,3] = 0.05

xbox = meshcat.geometry.Box([0.1, 0.01, 0.01])
viz.viewer["xbox1"].set_object(xbox, meshcat.geometry.MeshLambertMaterial(
                                 color=0xFF0000))
ybox = meshcat.geometry.Box([0.01, 0.1, 0.01])
viz.viewer["ybox1"].set_object(ybox, meshcat.geometry.MeshLambertMaterial(
                                 color=0x00FF00))
zbox = meshcat.geometry.Box([0.01, 0.01, 0.1])
viz.viewer["zbox1"].set_object(zbox, meshcat.geometry.MeshLambertMaterial(
                                 color=0x0000FF))

@dataclass
class ArmMeasurementCurrent:
    arm_orientation : np.array
    palm_orientation : np.array

imus = ArmMeasurement("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2")
estimate_x0 = np.zeros(rmodel.nq + rmodel.nv)

T = 30
measurement = collections.deque(maxlen=T)
counter = 0

Ori = np.eye(4)

print("reading measurements")
while True:
    imus.update_measurements()  
    measurement.append(ArmMeasurementCurrent(imus.arm_orientation, imus.palm_orientation))
    if counter > T:
        estimate = solve_estimation_problem(measurement, T, rmodel, estimate_x0)
        viz.display(estimate.xs[-1][:rmodel.nq])
        estimate_x0 = estimate.xs[-1]
        
        Ori[0:3,0:3] = imus.palm_orientation
        viz.viewer["xbox1"].set_transform(Ori @ X_TG)
        viz.viewer["ybox1"].set_transform(Ori @ Y_TG)
        viz.viewer["zbox1"].set_transform(Ori @ Z_TG)

    time.sleep(0.01)
    counter += 1

del imus