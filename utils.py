import meshcat
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import time 

# This class takes the current imu orienation and computes the arm and shoulder orinetation in the world frame
@dataclass
class ArmMeasurementCurrent:
    arm_orientation : np.array
    palm_orientation : np.array
    joint_angle : np.array

def add_frame(name, vis):
    xbox = meshcat.geometry.Box([0.1, 0.01, 0.01])
    vis["xbox_" + name].set_object(xbox, meshcat.geometry.MeshLambertMaterial(
                                    color=0xFF0000))
    ybox = meshcat.geometry.Box([0.01, 0.1, 0.01])
    vis["ybox_" + name].set_object(ybox, meshcat.geometry.MeshLambertMaterial(
                                    color=0x00FF00))
    zbox = meshcat.geometry.Box([0.01, 0.01, 0.1])
    vis["zbox_" + name].set_object(zbox, meshcat.geometry.MeshLambertMaterial(
                                    color=0x0000FF))

def update_frame(name, vis, R, offset = np.zeros(3)):
    X_TG = np.eye(4)
    X_TG[0,3] = 0.05
    Y_TG = np.eye(4)
    Y_TG[1,3] = 0.05
    Z_TG = np.eye(4)
    Z_TG[2,3] = 0.05

    offset_TG = np.eye(4)
    offset_TG[0:3,3] = offset


    T = np.eye(4)
    T[0:3,0:3] = R
    vis["xbox_" + name].set_transform( offset_TG @ T @ X_TG )
    vis["ybox_" + name].set_transform( offset_TG @ T @ Y_TG )
    vis["zbox_" + name].set_transform( offset_TG @ T @ Z_TG )




def visualize_estimate(child, viz):
    viewer = meshcat.Visualizer(zmq_url = "tcp://127.0.0.1:6014")
    viz.initViewer(viewer)
    viz.loadViewerModel()
    viz.initializeFrames()
    viz.display_frames = True
    add_frame("hand", viz.viewer)
    add_frame("shoulder", viz.viewer)
    while True:
        shoulder, hand, estimate = child.recv()
        update_frame("shoulder", viz.viewer,   shoulder)
        update_frame('hand', viz.viewer, hand, [0.5, 0, 0])
        viz.display(estimate[:5])
        


def visualize_solution(viz, child):
    viewer = meshcat.Visualizer(zmq_url = "tcp://127.0.0.1:6014")
    viz.initViewer(viewer)
    viz.loadViewerModel()
    viz.initializeFrames()
    viz.display_frames = True
    # add_frame("hand", viz.viewer)
    # add_frame("shoulder", viz.viewer)
    while True:
        xs = child.recv()
        for i in range(len(xs)):
            viz.display(xs[i][:5])
            time.sleep(0.001)
