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

class ResidualModelEnergy(crocoddyl.ResidualModelAbstract):
    def __init__(self, state):

        self.nq, self.nv = state.nq, state.nv        
        crocoddyl.ResidualModelAbstract.__init__(self, state, self.nv, True, True, True)
    
    def calc(self, data, x, u):
        v = x[self.nq:]
        for i in range(self.nv):
            data.r[i] = v[i]*u[i]
        # print(data.r)

    def calcDiff(self, data, x, u):
        v = x[self.nq:]
        data.Rx[:,self.nv:] = np.diag(-u)       
        data.Ru = np.diag(-v)


class DifferentialKinematicModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costModel, constraints = None):
        if constraints:
            crocoddyl.DifferentialActionModelAbstract.__init__(
                self, state, state.nv, costModel.nr, constraints.ng
            )
        else:
            crocoddyl.DifferentialActionModelAbstract.__init__(
                self, state, state.nv, costModel.nr
            )            
        self.costs = costModel
        self.nx = state.nq + state.nv
        self.A = np.zeros((state.nv, self.nx))
        self.B = np.eye(state.nv)       
        self.constraints = constraints

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[: self.state.nq], x[-self.state.nv :]

        pin.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
        # Computing the cost value and residuals
        data.xout = u
        pin.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pin.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost
        if self.constraints:
            data.constraints.resize(self, data)
            self.constraints.calc(data.constraints, x, u)

    def calcDiff(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        if u is None:
            u = self.unone
        if True:
            self.calc(data, x, u)
        # Computing the dynamics derivatives
        pin.computeRNEADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, data.xout
            )
        data.Fx = self.A
        data.Fu = self.B
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)
        if self.constraints:
            self.constraints.calcDiff(data.constraints, x, u)
    
    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pin.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        ) 
        # this allows us to share the memory of cost-terms of action model
        if self.constraints:
            data.constraints = self.constraints.createData(data.multibody)
            data.constraints.shareMemory(data)
        return data

class DifferentialFwdDynamicsReg(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, state.nv, costModel.nr
        )
        self.costs = costModel
        self.enable_force = True
        self.armature = np.zeros(0)

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[: self.state.nq], x[-self.state.nv :]
        # Computing the dynamics using ABA or manually for armature case
        if self.enable_force:
            data.xout = pin.aba(self.state.pinocchio, data.pinocchio, q, v, u)
        else:
            pin.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            data.M = data.pinocchio.M
            if self.armature.size == self.state.nv:
                data.M[range(self.state.nv), range(self.state.nv)] += self.armature
            data.Minv = np.linalg.inv(data.M) + 0.1*np.eye(self.state.nq)
            data.xout = data.Minv * (u - data.pinocchio.nle)
        # Computing the cost value and residuals
        pin.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pin.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        if u is None:
            u = self.unone
        if True:
            self.calc(data, x, u)
        # Computing the dynamics derivatives
        if self.enable_force:
            pin.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, u
            )
            data.Fx = np.hstack([data.pinocchio.ddq_dq, data.pinocchio.ddq_dv])
            data.Fu = data.pinocchio.Minv + 0.1*np.eye(self.state.nq)
        else:
            pin.computeRNEADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, data.xout
            )
            data.Fx = -np.hstack(
                [data.Minv * data.pinocchio.dtau_dq, data.Minv * data.pinocchio.dtau_dv]
            )
            data.Fu = data.Minv
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print("The armature dimension is wrong, we cannot set it.")
        else:
            self.enable_force = False
            self.armature = armature.T

    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pin.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        )  # this allows us to share the memory of cost-terms of action model
        return data


def solve_reaching_problem(x_des, x0, rmodel, T, dt, xs = None, us = None):
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
    activation = crocoddyl.ActivationModelWeightedQuad(np.array([0.01, 0.01, 0.01, 0.5, 0.01, 0.1, 0.1, 0.1, 0.1, 0.5 ]))
    xreg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
    xResidual = crocoddyl.ResidualModelState(state, xreg)
    xRegCost = crocoddyl.CostModelResidual(state, activation, xResidual)

    # energy cost
    energyResidual = ResidualModelEnergy(state)
    activation = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 1.0, 1.0, 1.0 ]))
    energyCost = crocoddyl.CostModelResidual(state, activation, energyResidual)

    # endeff frame translation cost
    endeff_frame_id = rmodel.getFrameId("Hand")
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, x_des)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    acc_refs = crocoddyl.ResidualModelJointAcceleration(state, nu)
    accCost = crocoddyl.CostModelResidual(state, acc_refs)    
    
    

    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 2e-4)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    # runningCostModel.addCost("translation", frameTranslationCost.copy(), 1e-3)
    # runningCostModel.addCost("acceleration", accCost, 1e-2)
    # runningCostModel.addCost("energy", energyCost, 1e-5)
    # terminalCostModel.addCost("stateReg", xRegCost, 5e-3)
    terminalCostModel.addCost("translation", frameTranslationCost.copy(), 1e1*dt)
    # terminalCostModel.addCost("acceleration", accCost, 5e-2)
    terminalCostModel.addCost("stateReg", xRegCost, 5e-3)


    #Constraints 
    constraints = crocoddyl.ConstraintModelManager(state, nu)
    ee_contraint = crocoddyl.ConstraintModelResidual(
    state,
    xResidual,
    np.array([-np.pi/2.0,-1000, -np.pi/2.0 , -np.pi/2.0, -np.pi/2.0,  -1.2, -1.2, -1.2, -1.2, -1.2]),
    np.array([0.1,1000, np.pi/2.0,  np.pi/2.0, np.pi,       1.2, 1.2, 1.2, 1.2, 1.2]),
)
    constraints.addConstraint("ee_bound", ee_contraint)

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    init_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel, constraints)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

    # init_DAM = DifferentialKinematicModel(state, runningCostModel)
    # running_DAM = DifferentialKinematicModel(state, runningCostModel, constraints)
    # terminal_DAM = DifferentialKinematicModel(state, terminalCostModel, constraints)
    
    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    initModel = crocoddyl.IntegratedActionModelEuler(init_DAM, dt)
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, 1*[initModel] + [runningModel] * (T-1), terminalModel)
    
    # Create solver + callbacks
    ddp = SolverCSQP(problem)
    ddp.setCallbacks([crocoddyl.CallbackLogger()])
    ddp.use_filter_line_search = True
    ddp.verbose = True
    ddp.with_callbacks = False
    # ddp.termination_tolerance = 1e-4
    ddp.eps_abs = 1e-4
    ddp.eps_rel = 1e-4
    ddp.max_qp_iters = 500
    # Warm start : initial state + gravity compensation
    # xinit = np.zeros(rmodel.nq + rmodel.nv)
    xinit = x0
    # print(xinit)
    if xs:
        xs_init = xs
        us_init = us
    else:
        xs_init = [xinit for i in range(T+1)]
        us_init = ddp.problem.quasiStatic(xs_init[:-1])
    
    # Solve
    ddp.solve(xs_init, us_init, maxiter=8)
    print(ddp.KKT)
    return ddp


def solve_reaching_problem_parallel(child_conn, T, dt):
    xs_prev = 0
    us_prev = 0
    while True:
        x_des, q0, rmodel = child_conn.recv()
        st = time.time()
        ddp = solve_reaching_problem(x_des, q0, rmodel, T, dt, xs_prev, us_prev)
        et = time.time()
        # print("ddp solve time : ", 1e3 * (et - st))
        xs_prev, us_prev = ddp.xs, ddp.us
        child_conn.send([np.array(ddp.xs), np.array(ddp.us)])