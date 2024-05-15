## This file contains a 5DoF arm model build from pinocchio
## Author : Avadesh Meduri
## Date : 29/02/2024

import pinocchio as pin
import numpy as np
import hppfcl as fcl

def create_arm():

    parent_id = 0
    arm_mass = 0.7
    upper_arm_radius = 2e-2
    upper_arm_length = 0.25
    lower_arm_radius = 1e-2
    lower_arm_length = 0.25
    axis_length = 0.1 # length of the co-ordinate axis rectangle


    rmodel = pin.Model()
    rmodel.name = "Human Arm"
    gmodel = pin.GeometryModel()

    # Joint
    joint_name = "Shoulder"
    Base_placement = pin.SE3.Identity()
    Base_id = rmodel.addJoint(parent_id, pin.JointModelSphericalZYX(), Base_placement, joint_name)

    frame_name = "imu_arm"
    imu_arm_placement = pin.SE3.Identity()
    imu_arm_placement.rotation = pin.utils.rpyToMatrix(np.pi/2.0,0,0) @ pin.utils.rpyToMatrix(0,0,-np.pi/2.0)
    imu_arm_placement.translation[2] = -upper_arm_length/2.0
    imu_arm_frame = pin.Frame(frame_name, Base_id, parent_id, imu_arm_placement, pin.OP_FRAME)
    rmodel.addFrame(imu_arm_frame)

    joint_name = "lower_arm_rotation"
    lar = pin.SE3.Identity()
    lar.rotation = pin.utils.rpyToMatrix(0,0,np.pi/2.0)
    lar.translation[2] = -lower_arm_length/2
    lar_id = rmodel.addJoint(Base_id, pin.JointModelRZ(), lar, joint_name)

    joint_name = "Elbow"
    elbow_placement = pin.SE3.Identity()
    elbow_placement.translation[2] = -upper_arm_length/2.0
    elbow_id = rmodel.addJoint(lar_id, pin.JointModelRY(), elbow_placement, joint_name)

    frame_name = "Hand"
    hand_placement = pin.SE3.Identity()
    hand_placement.rotation = pin.utils.rpyToMatrix(0,np.pi/2.0,0.0) @ pin.utils.rpyToMatrix(np.pi/2.0,0.0, 0)
    hand_placement.translation[2] = -lower_arm_length
    hand_frame = pin.Frame(frame_name, elbow_id, lar_id, hand_placement, pin.OP_FRAME)
    hand_id = rmodel.addFrame(hand_frame)

    # Upper Arm
    upper_arm1_inertia = pin.Inertia.FromCylinder(arm_mass/2.0, upper_arm_radius, upper_arm_length/2.0)
    upper_arm1_placement = Base_placement.copy()
    upper_arm1_placement.translation[2] = -upper_arm_length/4.0
    rmodel.appendBodyToJoint(Base_id, upper_arm1_inertia, upper_arm1_placement)

    geom_name = "Upper Arm1"
    shape = fcl.Cylinder(upper_arm_radius, upper_arm_length/2.0)
    shape_placement = upper_arm1_placement.copy()

    geom_obj = pin.GeometryObject(geom_name, Base_id, shape, shape_placement)
    geom_obj.meshColor = np.array([1.,1.,1.,1.])
    gmodel.addGeometryObject(geom_obj)

    # Upper Arm2
    upper_arm2_inertia = pin.Inertia.FromCylinder(arm_mass/2, lower_arm_radius, upper_arm_length/2)
    upper_arm2_placement = lar.copy()
    upper_arm2_placement.translation[2] = -lower_arm_length/4.0
    rmodel.appendBodyToJoint(lar_id, upper_arm2_inertia, upper_arm2_placement)

    geom_name = "Upper Arm2"
    shape = fcl.Cylinder(lower_arm_radius, lower_arm_length/2.0)
    shape_placement = upper_arm2_placement.copy()

    lower_arm_obj = pin.GeometryObject(geom_name, lar_id, shape, shape_placement)
    lower_arm_obj.meshColor = np.array([1.,1.,0.,1.])
    gmodel.addGeometryObject(lower_arm_obj)

    #Lower Arm2
    lower_arm_inertia = pin.Inertia.FromCylinder(arm_mass, lower_arm_radius, lower_arm_length)
    lower_arm_placement = elbow_placement.copy()
    lower_arm_placement.translation[2] = -lower_arm_length/2.0
    rmodel.appendBodyToJoint(elbow_id, lower_arm_inertia, lower_arm_placement)

    geom_name = "Lower Arm2"
    shape = fcl.Cylinder(1.5*lower_arm_radius, lower_arm_length)
    shape_placement = lower_arm_placement.copy()

    lower_arm_obj2 = pin.GeometryObject(geom_name, elbow_id, shape, shape_placement)
    lower_arm_obj2.meshColor = np.array([1.,0.,0.,1.])
    gmodel.addGeometryObject(lower_arm_obj2)


    # Hand 
    shape_base = fcl.Box(0.05,0.05,0.05)
    placement = pin.SE3.Identity()
    placement.translation[2] = -lower_arm_length - 0.05/2.0
    hand = pin.GeometryObject("hand", hand_id, elbow_id, shape_base, placement)
    hand.meshColor = np.array([0.,1.0,1.0,1.])
    gmodel.addGeometryObject(hand)

    cmodel = gmodel
    rdata = rmodel.createData()
    gdata = gmodel.createData()


    return rmodel, rdata, gmodel, cmodel