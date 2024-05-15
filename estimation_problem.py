import crocoddyl
from mim_solvers import SolverSQP
import numpy as np
import time


def solve_estimation_problem(measurements, T, rmodel, x0, activate_wts):
    rdata = rmodel.createData()
    nq = rmodel.nq; nv = rmodel.nv; nu = nq; nx = nq+nv

    
    dt = 1e-2
    
    # # # # # # # # # # # # # # #
    ###  SETUP CROCODDYL OCP  ###
    # # # # # # # # # # # # # # #
    
    # State and actuation model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    
    # Create cost terms 
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)

    runningModel = []
    for i in range(T):
        # endeff frame orientation cost
        endeff_frame_id = rmodel.getFrameId("Hand")
        frameOrientationResidual = crocoddyl.ResidualModelFrameRotation(state, endeff_frame_id, measurements[i].palm_orientation)
        frameOrientationCost = crocoddyl.CostModelResidual(state, frameOrientationResidual)

        imu_arm_id = rmodel.getFrameId("imu_arm")
        imuArmOrientationResidual = crocoddyl.ResidualModelFrameRotation(state, imu_arm_id, measurements[i].arm_orientation)
        imuArmOrientationCost = crocoddyl.CostModelResidual(state, imuArmOrientationResidual)
        
        # Running and terminal cost models
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)
        

        # State regularization cost
        x0[1] = measurements[i].joint_angle
        activation = crocoddyl.ActivationModelWeightedQuad(np.array([0.2, activate_wts*1e5, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3 ]))
        xResidual = crocoddyl.ResidualModelState(state, x0)
        xRegCost = crocoddyl.CostModelResidual(state, activation, xResidual)

        # Add costs
        runningCostModel.addCost("stateReg", xRegCost, 5e-2)
        runningCostModel.addCost("ctrlRegGrav", uRegCost, 5e-3)
        runningCostModel.addCost("shoulderOrientation", imuArmOrientationCost, 1e2)
        runningCostModel.addCost("wristOrientation", frameOrientationCost, 1e2)
        
    
        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
        runningModel.append(crocoddyl.IntegratedActionModelEuler(running_DAM, dt))

    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    
    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModel, terminalModel)

    # Create solver + callbacks
    ddp = SolverSQP(problem)
    # ddp = crocoddyl.SolverDDP(problem)
    ddp.setCallbacks([crocoddyl.CallbackLogger()])
    ddp.use_filter_line_search = True
    
    # Warm start : initial state + gravity compensation
    xs_init = [x0 for i in range(T+1)]
    us_init = ddp.problem.quasiStatic(xs_init[:-1])
    # Solve
    ddp.solve(xs_init, us_init, maxiter=10)
    return ddp


def solve_estimation_parallel(child_conn):
    while True:
        measurements, T, rmodel, x0, activate_wts = child_conn.recv()
        st = time.time()
        xs = solve_estimation_problem(measurements, T, rmodel, x0, activate_wts).xs
        et = time.time()
        # print("ddp solve time : ", 1e3 * (et - st))

        child_conn.send(np.array(xs[-1]))