import socket
import struct
import numpy as np
import time
import threading

MOTION_CMD = 0x01
MOTION_STATE = 0x01

class ExoSkeletonUDPInterface():
    def __init__(self, leg_ip='192.168.123.10', leg_port=5000, 
                       host_ip='192.168.123.1', host_port=5000,
                       num_actuators=1):
        self.num_actuators= num_actuators
        self.num_imus = 3
        self.leg_ip = leg_ip
        self.leg_port = leg_port
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.RX_RUNNING = True
        self.transmit_counter = 0
        self.socket.bind((host_ip, host_port))
        self.rx_thread=threading.Thread(target=self.receivingThread)
        self.rx_thread.start()
        self.state = None
        self.latest_state_stamp = time.time()
        self.is_calibrated = 0
        self.q_offsets = [0.]

    def calibrate(self):
        print("calibration ...")

        self.sendCommand([0.], [0.], [0.], [0.0], [0.0], [0.0])
        time.sleep(0.01)
        time_check = 0
        wait_time = 1.0

        while self.is_calibrated == 0:
            time.sleep(0.001)
            state = self.getState()
            if state is not None:
                data = self.getState()
                q_motor = data['motor_q']
                dq_motor  = data['motor_dq']
                # reset
                torque = 2.0
                self.sendCommand([0], [0.], [0], [0], [torque], [0.])
                if dq_motor < 1e-2:
                    time_check += 0.001
                if time_check > wait_time:
                    self.is_calibrated = 1
                    self.q_offsets = [q_motor,]
                    print("setting offset ...")
            else:
                print('No response received from the leg. Check the network.')

        print("finished calibration ...")
        self.sendCommand([0], [0.], [0], [0], [0.0], self.q_offsets)


    def setCommand(self,q_des, dq_des, kp, kd, tau_ff):
        if self.is_calibrated:
            if self.state['base_acc'] < 2 or self.state['shoulder_acc'] < 2 or self.state['wrist_acc'] < 2:
                print("please calibrate IMU to continue ...") 
            self.sendCommand(q_des, dq_des, kp, kd, tau_ff, self.q_offsets)
        else:
            print("Please calibrate the robot before sending command ....")
            self.sendCommand([0], [0.], [0], [0], [0.], [0.])

    def sendCommand(self,q_des, dq_des, kp, kd, tau_ff, q_offsets):
        assert self.num_actuators == len(q_des) == len(dq_des) == len(kp) == len(kd) == len(tau_ff) == len(q_offsets), 'there should be one entry in all inputs per actuator'
        msg_format=f'I{self.num_actuators*6}f'
        stamp = time.time()
        data = np.vstack([q_des, dq_des, kp, kd, tau_ff, q_offsets]).T.reshape(-1)
        arguments =  [msg_format] + [MOTION_CMD] + \
                      data.tolist()
        data=struct.pack(*arguments)
        self.socket.sendto(data, (self.leg_ip, self.leg_port))

    def getState(self):
        if self.state is not None and (time.time()-self.latest_state_stamp) < 0.1:
            return self.state
        else:
            return None
        

    def receivingThread(self):
        while self.RX_RUNNING:
            data = self.socket.recvmsg(4096)
            if data[0][0] == MOTION_STATE:
                msg_format=f'{self.num_actuators*(4) + (4+1)*(self.num_imus)}f'
                data = struct.unpack(msg_format, data[0][1:])
                state = np.array(data).reshape(self.num_actuators, 4 + (4+1)*(self.num_imus))  
                self.state={'q': state[:,0],
                            'dq': state[:,1],
                            'motor_q': state[:,2],
                            'motor_dq': state[:,3],
                            'base_ori': state[:,4:8],
                            'shoulder_ori': state[:,8:12],
                            'wrist_ori': state[:,12:16],
                            'base_acc': state[:,16],
                            'shoulder_acc': state[:,17],
                            'wrist_acc': state[:,18],}
                self.latest_state_stamp = time.time()
            else:
                print("invalid motion response")

    def terminate(self):
        self.RX_RUNNING = False
        self.socket.close()
        self.rx_thread.join()