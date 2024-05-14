import meshcat
from utils import *
import time 
import collections

from utils import ArmMeasurementCurrent, visualize_estimate, visualize_solution
from arm_model import create_arm
from pinocchio.visualize import MeshcatVisualizer as Visualizer
import meshcat
from estimation_problem import solve_estimation_parallel
from reaching_problem import solve_reaching_problem_parallel

from multiprocessing import Pipe, Process
import threading

from interface import ExoSkeletonUDPInterface

interface = ExoSkeletonUDPInterface()
counter = 0
interface.calibrate()

offset = [pin.utils.rpyToMatrix(np.pi/2.0, 0, 0), 0]

rmodel, rdata, gmodel, cmodel = create_arm()

q0 = np.array([0,-np.pi/6.0,0,-np.pi/2.0,0])

x_des = np.array([0.4, 0, -0.5])

data_offset = []

T_estimate = 10
estimate_x0 = np.zeros(rmodel.nq + rmodel.nv)
measurement = collections.deque(maxlen=T_estimate)

print("calibrating IMUS")
time.sleep(3)
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

# estimation threading setup
get_new_measurement = 1.0
estimate_parent, estimate_child = Pipe()
estimation_thread = Process(target = solve_estimation_parallel, args = (estimate_child,))
estimation_thread.start()

# mpc threading setup
solve_mpc = 1.0
T_mpc = 10
dt = 1e-1
mpc_parent, mpc_child = Pipe()
mpc_thread = Process(target = solve_reaching_problem_parallel, args = (mpc_child, T_mpc, dt))
mpc_thread.start()

# visualization threading setup
viz = Visualizer(rmodel, gmodel, cmodel)
viz_parent, viz_child = Pipe()
viz_thread = Process(target = visualize_solution, args = (viz, viz_child,))
viz_thread.start()

viz_estimate_parent, viz_estimate_child = Pipe()
viz_estimate_thread = Process(target = visualize_estimate, args = (viz_estimate_child, viz))
viz_estimate_thread.start()

for i in range (T_estimate):
    interface.setCommand([0], [0.], [0], [0], [1.0])
    time.sleep(0.001)
    state = interface.getState()
    state = interface.getState()
    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()    
    measurement.append(ArmMeasurementCurrent( base.T @ shoulder, base.T @ hand))

shoulder_offset_time = 500

offset_x0 = 0
offset_shoulder = 0

for i in range(shoulder_offset_time):
    interface.setCommand([0], [0.], [0], [0], [1.0])
    time.sleep(0.001)
    state = interface.getState()
    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()  
    estimate_parent.send([measurement, T_estimate, rmodel, np.zeros(rmodel.nq + rmodel.nv)])
    offset_x0 += estimate_parent.recv()[1]   

offset_shoulder = state["q"] + (offset_x0/shoulder_offset_time)


counter = 0
index = 0
ctrl_dt = 0.002
replan_freq = 0.01
knot_points = int(dt/ctrl_dt)
us = np.zeros((T_mpc, rmodel.nv))
buffer = 0.04/1e3

# statistics data 
ctrl_data = []
data_estimate = []
data_motor = []
data_torque = []

gst = time.perf_counter()
iteration_count = int(8e3)
no_torque = 0
interface.setCommand([0], [0.], [0], [0], [0.0])
time.sleep(0.001)

for i in range(iteration_count):
    st = time.perf_counter()
    state = interface.getState()

    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()    
    measurement.append(ArmMeasurementCurrent(imu_offset[0] @ base.T @ shoulder,imu_offset[1] @ base.T @ hand))

    if get_new_measurement:
        viz_estimate_parent.send([base.T @ shoulder, base.T @ hand, estimate_x0])
        estimate_parent.send([measurement, T_estimate, rmodel, np.zeros(rmodel.nq + rmodel.nv)])
        get_new_measurement = 0

    if estimate_parent.poll():
        estimate_x0 = estimate_parent.recv()   
        get_new_measurement = 1.0

    if solve_mpc and index*dt >= replan_freq:
        mpc_parent.send([x_des, estimate_x0[:rmodel.nq], rmodel])
        solve_mpc = 0

    if mpc_parent.poll():
        xs, us = mpc_parent.recv()         
        solve_mpc = 1.0
        viz_parent.send(xs)
        index = 0

    if counter % knot_points == 0:
        counter = 0
        index += 1
        torque_command_arr = np.linspace(us[index], us[index+1], endpoint = True, axis = 0, num = int(knot_points))

    # print(counter, knot_points, index, replan_freq)
    # print(i, estimate_x0[1], state['q'])
    data_estimate.append(estimate_x0[1])
    data_motor.append(-state["q"][0])
    torque_command = max(0.0,-0.2*torque_command_arr[counter][1])
    data_torque.append(torque_command_arr[counter][1])
    if i < 2000:
        torque_command = 0
    interface.setCommand([0], [0.], [0], [0], [torque_command])

    et = time.perf_counter()
    while (et - st) < ctrl_dt - buffer:
        time.sleep(0.000001)
        et = time.perf_counter()
    if i * ctrl_dt - (et - gst) > 1.0:
        print("Danger")
        assert False

    # print((et - st)*1e3)
    ctrl_data.append(index)
    counter += 1

get = time.perf_counter()

print("Actual total time :", get - gst)
print("Expected total time :", ctrl_dt * iteration_count)

import matplotlib.pyplot as plt

# plt.plot(data_estimate, label = "estimate")
# plt.plot(data_motor, label = "motor")
plt.plot(data_torque)
plt.grid()
plt.legend()
plt.show()

# control_loop(1,1, estimate_x0, counter, index)