import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import namedtuple
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
from attrdict import AttrDict
from numpy.linalg import inv
from numpy.linalg import pinv


MAX_EPISODE_LEN = 20*100

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('ori_initialize')
        # self.step_counter = 0
        p.connect(p.GUI)
        # p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        self.action_space = spaces.Box(np.array([0] * 9), np.array([3] * 9))
        self.observation_space = spaces.Box(np.array([-3] * 9), np.array([3] * 9))
        self._max_episode_steps = 1000
        self.ur5EndEffectorIndex = 11
        self.last_link_trn_end = None
        self.last_link_trn_pan = None
        print('ori_initialize')
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.pandaUid)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])
        joints = AttrDict()
        for i in range(numJoints):
            info = p.getJointInfo(self.pandaUid, i)
            jointID = info[0]
            jointName = info[1].decode('utf-8')
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity)
            joints[singleInfo.name] = singleInfo
            joint_name = ["x_dir_joint", "y_dir_joint", "z_dir_joint", "shoulder_pan_joint", "shoulder_lift_joint",
                          "elbow_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        num = 0
        for jointName in joints:
            if jointName in joint_name:
                joint = joints[jointName]
                parameter_sim = action[num]
                p.setJointMotorControl2(self.pandaUid, joint.id, p.VELOCITY_CONTROL,
                                        targetVelocity=parameter_sim,
                                        force=joint.maxForce)
                num = num + 1

        p.stepSimulation()

        # state_object = p.getJointState(self.pandaUid, 7)[0]
        state_object = p.getLinkState(self.pandaUid, 10)[0]
        # print("LinkState[10]: ",state_object[0])
        if state_object[1] > 0.2:
            # print(state_object)
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        info = {'object_position': state_object}

        xwb = p.getJointState(self.pandaUid, 0)[0]
        ywb = p.getJointState(self.pandaUid, 1)[0]
        qb = p.getJointState(self.pandaUid, 2)[0]
        q1 = p.getJointState(self.pandaUid, 5)[0]
        q2 = p.getJointState(self.pandaUid, 6)[0]
        q3 = p.getJointState(self.pandaUid, 7)[0]
        q4 = p.getJointState(self.pandaUid, 8)[0]
        q5 = p.getJointState(self.pandaUid, 9)[0]
        q6 = p.getJointState(self.pandaUid, 10)[0]

        state_robot = [xwb, ywb, qb, q1, q2, q3, q4, q5, q6]

        self.observation = state_robot

        done = False 
        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        print('mmp_reset')
        self.last_link_trn_end = None
        self.last_link_trn_pan = None
        self.step_counter = 0 #?
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        self.pandaUid = p.loadURDF("./MMP/urdf_bak/MMP.urdf")


        # generate 

        rest_poses = [0, 0, 0, 0.1308, -1.4659, -2.3576, 0.6814, 1.4407, -2.0944]
        p.resetJointState(self.pandaUid, 0, rest_poses[0])
        p.resetJointState(self.pandaUid, 1, rest_poses[1])
        p.resetJointState(self.pandaUid, 2, rest_poses[2])
        p.resetJointState(self.pandaUid, 5, rest_poses[3])
        p.resetJointState(self.pandaUid, 6, rest_poses[4])
        p.resetJointState(self.pandaUid, 7, rest_poses[5])
        p.resetJointState(self.pandaUid, 8, rest_poses[6])
        p.resetJointState(self.pandaUid, 9, rest_poses[7])
        p.resetJointState(self.pandaUid, 10, rest_poses[8])

        xwb = p.getJointState(self.pandaUid, 0)[0]
        ywb = p.getJointState(self.pandaUid, 1)[0]
        qb = p.getJointState(self.pandaUid, 2)[0]
        q1 = p.getJointState(self.pandaUid, 5)[0]
        q2 = p.getJointState(self.pandaUid, 6)[0]
        q3 = p.getJointState(self.pandaUid, 7)[0]
        q4 = p.getJointState(self.pandaUid, 8)[0]
        q5 = p.getJointState(self.pandaUid, 9)[0]
        q6 = p.getJointState(self.pandaUid, 10)[0]

        state_robot = [xwb, ywb, qb, q1, q2, q3, q4, q5, q6]

        self.observation = state_robot
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
    
    def getJointStates(self):
        robot = self.pandaUid
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def getMotorJointStates(self):
        robot = self.pandaUid
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        # print(len(joint_states))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        # print(len(joint_infos))
        # get valid joints

        joint_names = [i[1] for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        
        # print(joint_names)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    # def get_jacob(self):
    #     ur5EndEffectorIndex = 11
    #     # pos, vel, torq = self.getJointStates()
    #     mpos, mvel, mtorq = self.getMotorJointStates()
    #     print(mpos)
    #     result = p.getLinkState(self.pandaUid,
    #                     ur5EndEffectorIndex,
    #                     computeLinkVelocity=1,
    #                     computeForwardKinematics=1)
    #     link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        
    #     zero_acc = [0.0] * len(mpos)
    #     zero_vec = [0.0] * len(mpos)
    #     jv, jw = p.calculateJacobian(bodyUniqueId=self.pandaUid, linkIndex=ur5EndEffectorIndex,
    #                          localPosition=com_trn, objPositions=mpos,
    #                          objVelocities=zero_vec, objAccelerations=zero_acc)
    #     # jvv = (jv[:])[0:8]
    #     # jw = jw[0:8][:]
    #     print(np.array(jv)[:,0:9])
    #     # self.print_joint_info(5)
    #     return np.array(jv)[:,0:9],np.array(jw)[:,0:9]

    
    def get_jacob(self,end,frd):
        ur5EndEffectorIndex = end
        # pos, vel, torq = self.getJointStates()
        mpos, mvel, mtorq = self.getMotorJointStates()
        # print(mpos)
        result = p.getLinkState(self.pandaUid,
                        ur5EndEffectorIndex,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        
        zero_acc = [0.0] * len(mpos)
        zero_vec = [0.0] * len(mpos)
        jv, jw = p.calculateJacobian(bodyUniqueId=self.pandaUid, linkIndex=ur5EndEffectorIndex,
                             localPosition=com_trn, objPositions=mpos,
                             objVelocities=zero_vec, objAccelerations=zero_acc)
        # jvv = (jv[:])[0:8]
        # jw = jw[0:8][:]
        # print(np.array(jw)[:,0:9])
        jac = np.append(jv,jw,axis = 0)
        # self.print_joint_info(5)
        return np.array(jv)[:,0:9],np.array(jw)[:,0:9],np.append(np.array(jv)[:,0:9],np.array(jw)[:,0:9],axis = 0)[frd[0]:frd[1]+1,:]


    def print_joint_info(self,index):
        print("关节信息：")
        joint_info = p.getJointInfo(self.pandaUid, index)
        print(f"\
                [0]关节索引: {joint_info[0]}\n\
                [1]关节名称: {joint_info[1]}\n\
                [2]关节类型: {joint_info[2]}\n\
                [3]此主体的位置状态变量中的第一个位置索引: {joint_info[3]}\n\
                [4]在这个物体的速度状态变量中的第一个速度索引: {joint_info[4]}\n\
                [5]保留参数: {joint_info[5]}\n\
                [6]关节阻尼大小: {joint_info[6]}\n\
                [7]关节摩擦系数: {joint_info[7]}\n\
                [8]平动或转动关节的位置下限: {joint_info[8]}\n\
                [9]平动或转动关节的位置上限: {joint_info[9]}\n\
                [10]关节最大力矩: {joint_info[10]}\n\
                [11]关节最大速度: {joint_info[11]}\n\
                [12]连杆名称: {joint_info[12]}\n\
                [13]在当前连杆坐标系中表示的移动或转动的关节轴: {joint_info[13]}\n\
                [14]在父连杆坐标系中表示的关节位置: {joint_info[14]}\n\
                [15]在父连杆坐标系中表示的关节姿态(四元数x、y、z、w): {joint_info[15]}\n\
                [16]父连杆的索引，若是base连杆则返回-1: {joint_info[16]}\n\n")

        print(joint_info[1])  # b'Roll_Joint'
        print(joint_info[1].decode())  # Roll_Joint
        print(joint_info[1].decode("UTF-8"))  # Roll_Joint
        print(str(joint_info[12].decode()))  # Roll_Joint

    # input:position of 9 valid joints  
    def setJointPosition(self, position):
        robot = self.pandaUid
        p.setJointMotorControlArray(robot,
                                    [0,1,2,5,6,7,8,9,10],
                                    p.POSITION_CONTROL,
                                    targetPositions=position)
    
    # input:velocity of 9 valid joints  
    def setJointVelocity(self, velocity):
        robot = self.pandaUid
        p.setJointMotorControlArray(robot,
                                    [0,1,2,5,6,7,8,9,10],
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=velocity)
    
    def get_pan_position(self):
        # x = p.getJointState(self.pandaUid, 0)[0]
        # y = p.getJointState(self.pandaUid, 1)[0]
        result = p.getLinkState(self.pandaUid,
                        2,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        # print('link_trn',link_trn[0:2])
        return link_trn[0],link_trn[1]

    def get_end_position(self):
        result = p.getLinkState(self.pandaUid,
                        11,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        # print('link_trn',link_trn[0:2])
        return link_trn[0],link_trn[1],link_trn[2]

    def get_end_posture(self):
        result = p.getLinkState(self.pandaUid,
                        11,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        # print('link_rot',*link_rot)
        roll,pitch,yaw= quart_to_rpy(*link_rot)
        # print('r,p,y:',roll,pitch,yaw)
        return [roll,pitch,yaw]


    def draw_end_eff(self):
        result = p.getLinkState(self.pandaUid,
                        self.ur5EndEffectorIndex,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        if(self.last_link_trn_end==None):
            self.last_link_trn_end = link_trn
            pass
        else:
            p.addUserDebugLine(self.last_link_trn_end,link_trn,lineColorRGB=[0,0,1])
            self.last_link_trn_end = link_trn
    
    def get_joint_velocity(self):
        velocity = []
        joints = [0,1,2,5,6,7,8,9,10]
        for i,index in enumerate(joints):
            velocity.append(p.getJointState(self.pandaUid,index)[1])
        return velocity
    
    def get_joint_position(self):
        position = []
        joints = [6,7,8,9,10]
        for i,index in enumerate(joints):
            position.append(p.getJointState(self.pandaUid,index)[0])
        return position

    def draw_pan(self):
        result = p.getLinkState(self.pandaUid,
                        2,
                        # 6,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        link_trn = (np.array(link_trn)+np.array([0,0,0])).tolist()
        if(self.last_link_trn_pan==None):
            self.last_link_trn_pan = link_trn
            pass
        else:
            p.addUserDebugLine(self.last_link_trn_pan,link_trn,lineColorRGB=[1,0,0])
            self.last_link_trn_pan = link_trn

    def get_single_link_pos(self,index):
        """returns numpy array of the link position"""
        result = p.getLinkState(self.pandaUid,
                        index,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        return np.array(link_trn)

    def test(self):
        jv,jw,e = self.get_jacob(11)
        jac_t_pi = pinv(jv)
        jac = np.append(jv,jw,axis = 0)
        print(jac)
        jac_v = pinv(jac)
        ve = list(np.dot(jac_v, [0,0,0,0,0,0]))
        self.setJointVelocity(ve)
        self.draw_end_eff()
        p.stepSimulation()


def quart_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw