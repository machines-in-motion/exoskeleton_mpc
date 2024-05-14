## This file contains a simple shooting problem for reaching task of the hand
## This file is based on minimal examples crocoddyl
## Author : Avadesh Meduri
## Date : 29/02/2024

from mim_solvers import SolverSQP, SolverCSQP
import numpy as np
import pinocchio as pin
import crocoddyl
import time


def solve_reaching_problem(x_des, q0, rmodel, T, dt):
    rdata = rmodel.createData()
    nq = rmodel.nq; nv = rmodel.nv; nu = nq; nx = nq+nv
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0])
    pin.framesForwardKinematics(rmodel, rdata, q0)
    pin.computeJointJacobians(rmodel, rdata, q0)
    
    
    # # # # # # # # # # # # # # #
    ###  SETUP CROCODDYL OCP  ###
    # # # # # # # # # # # # # # #
    
    # State and actuation model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    
    # Running and terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    
    
    # Create cost terms 
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # endeff frame translation cost
    endeff_frame_id = rmodel.getFrameId("Hand")
    endeff_translation = x_des # move endeff +10 cm along x in WORLD frame
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    
    
    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-3)
    runningCostModel.addCost("translation", frameTranslationCost, 1e0)
    terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
    terminalCostModel.addCost("translation", frameTranslationCost, 1e2)
    
    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
    
    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    
    # Create solver + callbacks
    ddp = SolverSQP(problem)
    ddp.setCallbacks([crocoddyl.CallbackLogger()])
    ddp.use_filter_line_search = True
    
    # Warm start : initial state + gravity compensation
    xs_init = [x0 for i in range(T+1)]
    us_init = ddp.problem.quasiStatic(xs_init[:-1])
    
    # Solve
    ddp.solve(xs_init, us_init, maxiter=5)
    
    return ddp


def solve_reaching_problem_parallel(child_conn, T, dt):
    while True:
        x_des, q0, rmodel = child_conn.recv()
        st = time.time()
        ddp = solve_reaching_problem(x_des, q0, rmodel, T, dt)
        et = time.time()
        # print("ddp solve time : ", 1e3 * (et - st))
        child_conn.send([np.array(ddp.xs), np.array(ddp.us)])