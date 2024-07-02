import numpy as np
import collections
from utils import *

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer as Visualizer
import time
import meshcat
from arm_model import create_arm

from estimation_problem import solve_estimation_problem
from interface import ExoSkeletonUDPInterface
from utils import ArmMeasurementCurrent

print("This is the file")

interface = ExoSkeletonUDPInterface()
counter = 0
interface.calibrate()

offset = [0,0,0,1, 0]

rmodel, rdata, gmodel, cmodel = create_arm()

q0 = np.array([0,-np.pi/6.0,0,-np.pi/2.0,0])

data_offset = []

T_estimate = 10
estimate_x0 = np.zeros(rmodel.nq + rmodel.nv)
measurement = collections.deque(maxlen=T_estimate)
dt_estimate = 0.01

viz = Visualizer(rmodel, gmodel, cmodel)
viewer = meshcat.Visualizer(zmq_url = "tcp://127.0.0.1:6000")
viz.initViewer(viewer)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True


print("calibrating IMUS")
#calibrating IMUS
data_offset = []
imu_offset = [pin.utils.rpyToMatrix(np.pi/2.0, 0, 0), 0]
calibration_time = 100
for i in range (calibration_time):
    interface.setCommand([0], [0.], [0], [0], [0.0])
    time.sleep(0.001)
    state = interface.getState()

    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()
    data_offset.append([base.T @ shoulder, base.T @ hand])

imu_offset[0] = imu_offset[0] @ data_offset[50][0].T
imu_offset[1] = data_offset[50][1].T

for i in range (T_estimate):
    interface.setCommand([0], [0.], [0], [0], [2.0])
    time.sleep(0.001)
    state = interface.getState()

    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()    
    measurement.append(ArmMeasurementCurrent(imu_offset[0] @ base.T @ shoulder,imu_offset[1] @ base.T @ hand, 0))

shoulder_offset_time = 500

offset_x0 = [0,0]
motor_offset_shoulder = 0

print("estimating motor offset ...")

for i in range(shoulder_offset_time):
    interface.setCommand([0], [0.], [0], [0], [2.0])
    time.sleep(0.001)
    state = interface.getState()
    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()  
    measurement.append(ArmMeasurementCurrent(imu_offset[0] @ base.T @ shoulder,imu_offset[1] @ base.T @ hand, 0))
    estimate_x0 = solve_estimation_problem(measurement, T_estimate, dt_estimate, rmodel, 0).xs[-1]
    offset_x0[0] += estimate_x0[1]
    offset_x0[1] += state["q"][0]
motor_offset_shoulder = (offset_x0[0] - offset_x0[1])/shoulder_offset_time


for i in range (T_estimate):
    interface.setCommand([0], [0.], [0], [0], [2.0])
    time.sleep(0.001)
    state = interface.getState()
    joint_angle = state['q'][0] + motor_offset_shoulder

    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()    
    measurement.append(ArmMeasurementCurrent(imu_offset[0] @ base.T @ shoulder,imu_offset[1] @ base.T @ hand, joint_angle))

while True:
    interface.setCommand([0], [0.], [0], [0], [1.0])
    time.sleep(0.001)
    state = interface.getState()
    measurement.append(ArmMeasurementCurrent(imu_offset[0] @ base.T @ shoulder,imu_offset[1] @ base.T @ hand, joint_angle))
    estimate = solve_estimation_problem(measurement, T_estimate, dt_estimate, rmodel, 1)
    viz.display(estimate.xs[-1][:rmodel.nq])
    estimate_x0 = estimate.xs[-1]
    print(estimate_x0[1]- state["q"], state["motor_q"])
    # time.sleep(0.01)
    counter += 1

