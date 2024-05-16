import meshcat
from utils import *
import time 
import collections

from utils import ArmMeasurementCurrent, visualize_estimate, visualize_solution
from arm_model import create_arm
from pinocchio.visualize import MeshcatVisualizer as Visualizer
from estimation_problem import solve_estimation_parallel
from reaching_problem import solve_reaching_problem_parallel

from multiprocessing import Pipe, Process
import threading

from interface import ExoSkeletonUDPInterface

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
recieve_new_estimate = 0.0
estimate_parent, estimate_child = Pipe()
estimation_thread = Process(target = solve_estimation_parallel, args = (estimate_child,))
estimation_thread.start()

# mpc threading setup
solve_mpc = 1.0
recieve_new_measurement = 0.0
T_mpc = 10
dt = 0.1   
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
    estimate_parent.send([measurement, T_estimate, rmodel, np.zeros(rmodel.nq + rmodel.nv), 0])
    estimate_x0 = estimate_parent.recv()
    offset_x0[0] += estimate_x0[1]
    offset_x0[1] += state["q"][0]
    # print(estimate_parent.recv()[1], state["q"][0])
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

counter = 1
index = 0
ctrl_dt = 0.002
replan_freq = 0.1
knot_points = int(dt/ctrl_dt)
us = np.zeros((T_mpc, rmodel.nv))
buffer = 0.04/1e3

# statistics data 
effecto_estimate = []
data_estimate = []
data_motor = []
data_torque = []

gst = time.perf_counter()
iteration_count = int(2e4)
no_torque = 0
interface.setCommand([0], [0.], [0], [0], [0.0])
time.sleep(0.001)

# target
x_des = np.array([0.4, -0.2, -0.0])

# vicon_target = ViconTarget()

for i in range(iteration_count):
    st = time.perf_counter()
    state = interface.getState()
    joint_angle = state['q'][0] + motor_offset_shoulder

    base = Rotation.from_quat(state["base_ori"][0]).as_matrix()
    shoulder = Rotation.from_quat(state["shoulder_ori"][0]).as_matrix()
    hand = Rotation.from_quat(state["wrist_ori"][0]).as_matrix()    
    measurement.append(ArmMeasurementCurrent(imu_offset[0] @ base.T @ shoulder,imu_offset[1] @ base.T @ hand, joint_angle))

    if get_new_measurement:
        viz_estimate_parent.send([imu_offset[0] @ base.T @ shoulder, imu_offset[1] @ base.T @ hand, estimate_x0, x_des])
        estimate_parent.send([measurement, T_estimate, rmodel, np.zeros(rmodel.nq + rmodel.nv), 1])
        get_new_measurement = 0
        recieve_new_estimate = 1.0

    if estimate_parent.poll() and recieve_new_estimate:
        estimate_x0 = estimate_parent.recv()   
        get_new_measurement = 1.0
        recieve_new_estimate = 0.0

    if solve_mpc and index*dt >= replan_freq:
        # x_des = vicon_target.get_taget()
        mpc_parent.send([x_des.copy(), estimate_x0.copy(), rmodel])
        solve_mpc = 0
        recieve_new_measurement = 1.0

    if mpc_parent.poll() and recieve_new_measurement:
        xs, us = mpc_parent.recv()         
        solve_mpc = 1.0
        recieve_new_measurement = 0.0
        # viz_parent.send(xs)
        index = 0

    if (counter) % knot_points == 0:
        counter = 1
        index += 1
    if counter == 1:
        torque_command_arr = np.linspace(us[index][1], us[index+1][1], endpoint = True, num = int(knot_points))
    # print(joint_angle, estimate_x0[1])
    # print(counter, knot_points, index, replan_freq, recieve_new_measurement, solve_mpc)
    pin.framesForwardKinematics(rmodel, rdata, estimate_x0[:rmodel.nq])
    pin.updateFramePlacements(rmodel, rdata)
    data_estimate.append(estimate_x0[1])
    data_motor.append(joint_angle)
    effecto_estimate.append(np.array(rdata.oMf[rmodel.getFrameId("Hand")].translation).copy())

    tau_grav = pin.rnea(rmodel, rdata, estimate_x0[:rmodel.nq], np.zeros(rmodel.nv), np.zeros(rmodel.nv))[1]
    desired_joint_torque = torque_command_arr[counter-1]
    #TODO: jacobian should me moved to the firmware
    motor_torque_grav = (2*0.16600942* state["motor_q"][0] - 0.73458596)*tau_grav
    motor_torque = (2*0.16600942* state["motor_q"][0] - 0.73458596)*desired_joint_torque

    if i < 2000:
        torque_command = 0.3
    else:
        print("semdomg cpommand")
        torque_command = max(0.3, motor_torque)
    interface.setCommand([0], [0.], [0], [0], [torque_command])

    data_torque.append([desired_joint_torque, torque_command])

    et = time.perf_counter()
    while (et - st) < ctrl_dt - buffer:
        time.sleep(0.000001)
        et = time.perf_counter()
    if i * ctrl_dt - (et - gst) > 1.0:
        print("Danger")
        assert False

    counter += 1

get = time.perf_counter()

print("Actual total time :", get - gst)
print("Expected total time :", ctrl_dt * iteration_count)

interface.setCommand([0], [0.], [0], [0], [0.0])

import matplotlib.pyplot as plt

data_torque = np.array(data_torque)
effecto_estimate = np.array(effecto_estimate)

fig, ax = plt.subplots(3, sharex = True)


time_scale = ctrl_dt * np.arange(0, len(data_torque))
ax[0].plot(time_scale, data_torque[:,0], '--bo', label = "ddp")
ax[0].plot(time_scale, data_torque[:,1], label = "motor")
ax[0].grid()
ax[0].legend()

ax[1].plot(time_scale, (180/np.pi)*np.array(data_estimate), label = "estimate angle")
ax[1].plot(time_scale, (180/np.pi)*np.array(data_motor), label = "joint angle")
ax[1].grid()
ax[1].legend()



ax[2].plot(time_scale, effecto_estimate[:,2], label = "estimate")
ax[2].plot(time_scale, len(effecto_estimate)*[x_des[2]], label = "target")

# plt.plot(data_torque)
plt.grid()
plt.legend()
plt.show()

# control_loop(1,1, estimate_x0, counter, index)