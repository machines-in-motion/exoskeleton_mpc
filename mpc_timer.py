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

rmodel, rdata, gmodel, cmodel = create_arm()

q0 = np.array([0,-np.pi/6.0,0,-np.pi/2.0,0])

x_des = np.array([0.4, 0, 0,])

data_offset = []
offset = [pin.utils.rpyToMatrix(np.pi/2.0, 0, 0), 0]

T_estimate = 10
estimate_x0 = np.zeros(rmodel.nq + rmodel.nv)
measurement = collections.deque(maxlen=T_estimate)

# estimation threading setup
get_new_measurement = 1.0
estimate_parent, estimate_child = Pipe()
estimation_thread = Process(target = solve_estimation_parallel, args = (estimate_child,))
estimation_thread.start()

# mpc threading setup
solve_mpc = 1.0
T_mpc = 100
dt = 1e-2
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
    q = q0 + 0.1* np.random.rand(rmodel.nq)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)
    base = rdata.oMi[rmodel.getJointId("universe")].rotation
    shoulder = rdata.oMf[rmodel.getFrameId("imu_arm")].rotation
    hand = rdata.oMf[rmodel.getFrameId("Hand")].rotation
    measurement.append(ArmMeasurementCurrent( base.T @ shoulder, base.T @ hand))

estimate_parent.send([measurement, T_estimate, rmodel, estimate_x0])
estimate_x0 = estimate_parent.recv()   

counter = 0
index = 0
ctrl_dt = 0.002
knot_points = int(dt/ctrl_dt)
us = np.zeros((T_mpc, rmodel.nv))
buffer = 0.05/1e3

# statistics data 
ctrl_data = []


gst = time.perf_counter()
iteration_count = 6000

for i in range(iteration_count):
# def control_loop(get_new_measurement, solve_mpc, estimate_x0, counter, index):
    st = time.perf_counter()
    q = q0 + 0.1* np.random.rand(rmodel.nq)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)
    base = rdata.oMi[rmodel.getJointId("universe")].rotation
    shoulder = rdata.oMf[rmodel.getFrameId("imu_arm")].rotation
    hand = rdata.oMf[rmodel.getFrameId("Hand")].rotation
    measurement.append(ArmMeasurementCurrent( base.T @ shoulder, base.T @ hand))

    if get_new_measurement:
        # viz_estimate_parent.send([base.T @ shoulder, base.T @ hand])
        estimate_parent.send([measurement, T_estimate, rmodel, estimate_x0])
        get_new_measurement = 0

    if estimate_parent.poll():
        estimate_x0 = estimate_parent.recv()   
        get_new_measurement = 1.0

    if solve_mpc:
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
        torque_command_arr = np.linspace(us[index], us[index+1], endpoint = True, axis = 1, num = int(knot_points))

    torque_command = torque_command_arr[counter]

    et = time.perf_counter()
    while (et - st) < ctrl_dt - buffer:
        time.sleep(0.000001)
        et = time.perf_counter()

    if i * ctrl_dt - (et - gst) > 1.0:
        print("Danger")
        assert False
    # print((et - st)*1e3)
    ctrl_data.append((et - st)*1e3)
    counter += 1

get = time.perf_counter()

print("Actual total time :", get - gst)
print("Expected total time :", ctrl_dt * iteration_count)

import matplotlib.pyplot as plt

plt.plot(ctrl_data)
plt.show()

# control_loop(1,1, estimate_x0, counter, index)