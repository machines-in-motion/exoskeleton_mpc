import meshcat
from utils import *
from pinocchio.visualize import MeshcatVisualizer as Visualizer
from reaching_problem import solve_reaching_problem
from arm_model import create_arm
import time
import matplotlib.pyplot as plt

rmodel, rdata, gmodel, cmodel = create_arm()
q0 = np.array([0,-np.pi/6.0,0,0.0,-np.pi/8.0])

viz = Visualizer(rmodel, gmodel, cmodel)
viewer = meshcat.Visualizer(zmq_url = "tcp://127.0.0.1:6014")
viz.initViewer(viewer)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True
# add_frame("hand", viz.viewer)
# add_frame("shoulder", viz.viewer)

T = 10
dt = 0.1
x_des_arr = np.array([[0.2, -.3, 0.0], [0.3, 0, 0.3], [0.3, 0.3, -0.4],  [0.3, -0.3, -0.2]])
x0 = np.zeros(rmodel.nq + rmodel.nv)
x0[:rmodel.nq] = q0

tau_arr = []

for j in range(60):
    x_des = x_des_arr[int(j/20)]
    print(x_des)
    ddp = solve_reaching_problem(x_des, x0, rmodel, T, dt)
    xs, us = ddp.xs, ddp.us
    for i in range(len(us)):
        viz.display(xs[i][:5])
        tau_arr.append(us[i][1])
        time.sleep(0.01)
    x0 = xs[-1]


plt.plot(tau_arr)
plt.grid()
plt.show()