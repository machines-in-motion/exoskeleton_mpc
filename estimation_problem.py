import crocoddyl
from mim_solvers import SolverSQP, SolverCSQP
import numpy as np
import time
from reaching_problem import DifferentialKinematicModel


def solve_estimation_problem(measurements, T, dt, rmodel, activate_wts, xs = None):
    rdata = rmodel.createData()
    nq = rmodel.nq; nv = rmodel.nv; nu = nq; nx = nq+nv
    
    if not xs:
        xs = [np.zeros(nx) for i in range(T)]
    # # # # # # # # # # # # # # #
    ###  SETUP CROCODDYL OCP  ###
    # # # # # # # # # # # # # # #
    
    # State and actuation model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    
    # Create cost terms 
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControl(state)
    activation = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 1.0, 1.0, 1.0 ]))
    uRegCost = crocoddyl.CostModelResidual(state, activation, uResidual)

    runningModel = []
    for i in range(T):
        # endeff frame orientation cost
        endeff_frame_id = rmodel.getFrameId("Hand")
        frameOrientationResidual = crocoddyl.ResidualModelFrameRotation(state, endeff_frame_id, measurements[i].palm_orientation)
        frameOrientationCost = crocoddyl.CostModelResidual(state, frameOrientationResidual)

        imu_arm_id = rmodel.getFrameId("imu_arm")
        imuArmOrientationResidual = crocoddyl.ResidualModelFrameRotation(state, imu_arm_id, measurements[i].arm_orientation)
        imuArmOrientationCost = crocoddyl.CostModelResidual(state, imuArmOrientationResidual)

        # State regularization cost
        if i == T-1:
            xs[i] = np.zeros(nx)
            xs[i][1] = measurements[i].joint_angle

        xResidual = crocoddyl.ResidualModelState(state, xs[i])
        activation = crocoddyl.ActivationModelWeightedQuad(np.array([0.1, 0*5e1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ]))
        xRegCost = crocoddyl.CostModelResidual(state, activation, xResidual)

        #constraints
        constraints = crocoddyl.ConstraintModelManager(state, nu)
        ee_contraint = crocoddyl.ConstraintModelResidual(
        state,
        xResidual,
            np.array([-np.pi/2.0,-1000, -np.pi/2.0 ,    0.0, -np.pi/2.0,  -1.2, -1.2, -1.2, -1.2, -1.2]),
            np.array([0.1,        1000,   np.pi/2.0,  np.pi,     np.pi/2.0,       1.2, 1.2, 1.2, 1.2, 1.2]),
        )
        constraints.addConstraint("ee_bound", ee_contraint)

        if i != T-1:
            # Running and terminal cost models
            runningCostModel = crocoddyl.CostModelSum(state)

            # Add costs
            runningCostModel.addCost("stateReg", xRegCost, 5e-2)
            runningCostModel.addCost("shoulderOrientation", imuArmOrientationCost, 1e1)
            runningCostModel.addCost("wristOrientation", frameOrientationCost, 1e1)

            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            
            # running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel, constraints)
            running_DAM = DifferentialKinematicModel(state, runningCostModel, constraints)
            
            runningModel.append(crocoddyl.IntegratedActionModelEuler(running_DAM, dt))
        else:
            terminalCostModel = crocoddyl.CostModelSum(state)
            terminalCostModel.addCost("stateReg", xRegCost, 5e-2)
            terminalCostModel.addCost("shoulderOrientation", imuArmOrientationCost, 5e1)
            terminalCostModel.addCost("wristOrientation", frameOrientationCost, 5e1)

            # terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel, constraints)
            terminal_DAM = DifferentialKinematicModel(state, terminalCostModel, constraints)
            
            # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
            
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(xs[1], runningModel, terminalModel)

    # Create solver + callbacks
    if not activate_wts:
        ddp = SolverSQP(problem)
    else:
        ddp = SolverCSQP(problem)
        ddp.max_qp_iters = 50

    # ddp = crocoddyl.SolverDDP(problem)
    ddp.setCallbacks([crocoddyl.CallbackLogger()])
    ddp.use_filter_line_search = True
    ddp.with_callbacks = True
    ddp.termination_tolerance = 1e-2

    # Warm start : initial state + gravity compensation
    xs_init = xs
    us_init = [np.zeros(rmodel.nv) for i in range(T-1)]
    # Solve
    if not activate_wts:
        ddp.solve(xs_init, us_init, maxiter=15)
    else:
        ddp.solve(xs_init, us_init, maxiter=10)

    if ddp.KKT > 1e-1:
        print("Warning : estimation not converging")
    return ddp


def solve_estimation_parallel(child_conn):
    xs_prev = None
    while True:
        measurements, T, dt, rmodel, activate_wts = child_conn.recv()
        st = time.time()
        xs = solve_estimation_problem(measurements, T, dt, rmodel, activate_wts, xs_prev).xs
        et = time.time()
        # print("ddp solve time : ", 1e3 * (et - st))
        xs_prev = xs.copy()
        estimate = np.array(xs[-1])
        estimate[:rmodel.nq] = estimate[:rmodel.nq]%(2*np.pi)
        for i in range(rmodel.nq):
            if estimate[i] > np.pi:
                estimate[i] -= 2*np.pi
        child_conn.send(estimate)