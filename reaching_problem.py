## This file contains a simple shooting problem for reaching task of the hand
## This file is based on minimal examples crocoddyl
## Author : Avadesh Meduri
## Date : 29/02/2024

from mim_solvers import SolverSQP, SolverCSQP
from crocoddyl import SolverDDP
import numpy as np
import pinocchio as pin
import crocoddyl
import time


def solve_reaching_problem(x_des, x0, rmodel, T, dt, xs = None):
    nq = rmodel.nq; nv = rmodel.nv; nu = nq; nx = nq+nv
    assert len(x0) == nx
    q0, v0 = x0[:nq], x0[nq:]
        
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
    # uResidual = crocoddyl.ResidualModelControlGrav(state)
    uResidual = crocoddyl.ResidualModelControl(state)
    activation = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 1.0, 1.0, 1.0 ]))
    uRegCost = crocoddyl.CostModelResidual(state, activation, uResidual)
    # State regularization cost
    activation = crocoddyl.ActivationModelWeightedQuad(np.array([0.1, 0.1, 0.1, 0.01, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1 ]))
    xreg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
    xResidual = crocoddyl.ResidualModelState(state, xreg)
    xRegCost = crocoddyl.CostModelResidual(state, activation, xResidual)

    # endeff frame translation cost
    endeff_frame_id = rmodel.getFrameId("Hand")
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, x_des)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    acc_refs = crocoddyl.ResidualModelJointAcceleration(state, nu)
    accCost = crocoddyl.CostModelResidual(state, acc_refs)    
    
    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 2e-4)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-3)
    runningCostModel.addCost("translation", frameTranslationCost.copy(), 2e1)
    runningCostModel.addCost("acceleration", accCost, 1e-2)
    # terminalCostModel.addCost("stateReg", xRegCost, 5e-3)
    terminalCostModel.addCost("translation", frameTranslationCost.copy(), 2e1*dt)
    # terminalCostModel.addCost("acceleration", accCost, 5e-2)

    #Constraints 
    constraints = crocoddyl.ConstraintModelManager(state, nu)
    ee_contraint = crocoddyl.ConstraintModelResidual(
    state,
    xResidual,
    np.array([-np.pi/3,-np.pi/3,0.0, -1000, -np.pi/2, -0.1, -.1, -.1, -.1, -.1]),
    np.array([0, np.pi/3.0, np.pi/2, np.pi/2, 1000, 0.1, 0.1, .1, .1, .1]),
)
    constraints.addConstraint("ee_bound", ee_contraint)

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel, constraints)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
    
    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    
    # Create solver + callbacks
    ddp = SolverCSQP(problem)
    ddp.setCallbacks([crocoddyl.CallbackLogger()])
    ddp.use_filter_line_search = True
    ddp.verbose = True
    ddp.with_callbacks = False
    # ddp.termination_tolerance = 1e-4
    ddp.eps_abs = 1e-6
    ddp.eps_rel = 1e-6
    # Warm start : initial state + gravity compensation
    # xinit = np.zeros(rmodel.nq + rmodel.nv)
    xinit = x0
    if xs:
        xs_init = xs
    else:
        xs_init = [xinit for i in range(T+1)]
    us_init = ddp.problem.quasiStatic(xs_init[:-1])
    
    # Solve
    ddp.solve(xs_init, us_init, maxiter=10)
    
    return ddp


def solve_reaching_problem_parallel(child_conn, T, dt):
    while True:
        x_des, q0, rmodel = child_conn.recv()
        st = time.time()
        ddp = solve_reaching_problem(x_des, q0, rmodel, T, dt)
        et = time.time()
        # print("ddp solve time : ", 1e3 * (et - st))
        child_conn.send([np.array(ddp.xs), np.array(ddp.us)])