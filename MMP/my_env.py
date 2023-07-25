from MMP.MMP_env import PandaEnv as env1
from MMP.task_modulate import task_modulator
from MMP.traj_generator import Bezier,point_generator
import numpy as np
import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data
import math
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=120)
# np.core.arrayprint._line_width = 180
class my_env(gym.Env):
    def __init__(self):
        self.action_dim = 7
        self.plant = env1()
        self.m = task_modulator()
        self.action_space = spaces.Box(np.array([0] * self.action_dim), np.array([1] * self.action_dim))
        self.observation_space = spaces.Box(np.array([-3] * self.action_dim), np.array([3] * self.action_dim))
        self.plant.reset()
        self.obs_pos = []
        self.task1 = [0,0,0]
        self.pt_gen = point_generator()
        self.bz = Bezier(np.array([[0,0,0]]),2000)
        self.traj_pts = []
        self.traj_v = []
        self.control_pts = []
        self.des_pt = None
        self.des_pt_last = None
        self.des_posture = None
        self.draw_offset = None
        self.step_count = 0
        self.sim_time = 20
        self.num_points = 6000
        p.setRealTimeSimulation(0)  
        p.setTimeStep(1./240.)  
        self.traj_exactness = 0
        self.last_joint_v = []
        self.bound_joints = [6,7,8,9,10]
        self.bound = [(-3,2,-2.5,1.5),(-3,3,-2.5,2.5),(-3,3,-2.5,2.5),(-3,3,-2.5,2.5),(-3,3,-2.5,2.5)] 
        print('my_initialize')
        
        
    def step(self,action):
        task1,task2 =   self.generate_task()
        action = [1,1,1,0,0,0,0]
        # execute forward
        self.m.update_tasks(task1,task2)
        self.m.update_s(action)
        # print('x_s:',self.m.x_S)
        # print('s:',action)
        ve = self.m.merge(self.plant)
        robot_state,_,__,___ = self.plant.step(ve)

        # get observation
        self.observation = np.array(self.m.x_S).astype(np.float32)

        ################# get reward and done ##############################
        reward = 0
        done = False

        ################## collision penalty
        x,y = self.plant.get_pan_position()
        for pos in self.obs_pos:
            d = math.sqrt((x-pos[0])**2+(y-pos[1])**2)
            if d < 0.5:
                print('encounter collision')
                reward += -50
                done = True
                break
        ################## traj exactness
        delta = (self.des_pt[0] - self.plant.get_end_position()[0])**2+(self.des_pt[1] - self.plant.get_end_position()[1])**2+(self.des_pt[2] - self.plant.get_end_position()[2])**2
        # self.traj_exactness = (1 - 1/self.step_count)*self.traj_exactness - 1/self.step_count*delta
        
        # reward += self.traj_exactness
        # print('delta:',delta)
        # print('exactness:',self.traj_exactness)
        ################## effort penalty
        sum_acc = 0
        acc_threshold = 0.05
        ves = self.plant.get_joint_velocity()
        if len(self.last_joint_v) == 0:
            self.last_joint_v = ves
        else:
            for i,ve in enumerate(ves):
                if math.fabs(ve-self.last_joint_v[i]) > acc_threshold:
                    sum_acc += (ve - self.last_joint_v[i])**2
            self.last_joint_v = ves

        ################## eventual reward (no use)
        dt = 1.0/240.0
        i_v = int(self.step_count*dt/self.sim_time*self.num_points)
        if i_v > self.num_points - 2:
            done = True 
            reward += 20

        ################## joint bound penalty
        bot_id = self.plant.pandaUid
        for i,index in enumerate(self.bound_joints):
            q = p.getJointState(bot_id, index)[0]
            max = self.bound[i][1]
            min = self.bound[i][0]
            if q > max or q < min:
                reward += -30
                done = True 
                print('joint:',index,'q',q,'joint angle over boundary')
    
        # draw traj
        self.plant.draw_end_eff()
        self.plant.draw_pan()
        if self.des_pt_last == None:
            self.des_pt_last = self.des_pt
        else:
            p.addUserDebugLine(self.des_pt_last,self.des_pt,lineColorRGB=[0,1,0])
            self.des_pt_last = self.des_pt

        return self.observation,reward,done,{}
    

    def render(self):
        self.plant.render()
    
    def reset(self):
        ret = self.plant.reset()
        # self.obs_pos = self.generate_obs(10)
        # self.observation = np.array(self.m.x_S).astype(np.float32)     
        self.control_pts = self.pt_gen.get_points(100)
        self.bz = Bezier(self.control_pts,self.num_points)
        self.traj_v = self.bz.get_v(0,self.sim_time)
        # self.traj_v = self.bz.get_v_with_posture(0,self.sim_time)
        x0,y0,z0 = self.plant.get_end_position()
        self.traj_pts = (self.bz.getBezierPoints(0)+np.array([[x0,y0,z0]])).tolist()
        self.obs_pos = self.generate_obs_new_new(15)
        self.step_count = 0
        self.traj_exactness = 0
        self.des_pt_last = None
        self.des_posture = self.plant.get_end_posture()
        # print('des_posture',self.des_posture) 
        self.step(np.zeros(self.action_dim).tolist())
        self.draw_offset = self.plant.last_link_trn_end
        print('my_env reset')
        return np.array(self.m.x_S).astype(np.float32)
    
    def close(self):
        self.plant.close()

    def generate_task(self):
        # random sampling of main task
        dt = 1.0/240.0
        i_v = int(self.step_count*dt/self.sim_time*self.num_points)
        # print('step_count:',self.step_count)
        # print('i_v:',i_v)
        # print('len of v:',len(self.traj_v))
        self.task1 = self.traj_v[i_v].copy()
        # print('traj_v:',self.traj_v[i_v])
        # print('task1 1.1',self.task1)
        kp = 5
        end_pos = self.plant.get_end_position()
        now_posture = self.plant.get_end_posture()
        v_posture = []
        # self.plant.get_end_posture()
        if self.step_count>1:
            self.task1 = (np.array(self.task1) + kp*(np.array(self.des_pt) - np.array(end_pos))).tolist()
        v_posture = ((-np.array(now_posture)+np.array(self.des_posture))).tolist()
        self.step_count += 1
        # _,__,jac1 = self.plant.get_jacob(11,[0,2])
        _,__,jac1 = self.plant.get_jacob(11,[0,5])
        self.task1 = self.task1 + v_posture
        # print('task1 1.2',self.task1)
        # self.task1 = [0,0,0,0,0,2]
        task1 = (self.task1,jac1)
        self.des_pt = self.traj_pts[i_v]
        # print('task1_v',self.task1)

        
        # avoid obstacle
        delta = 0.0004
        bot_x,bot_y = self.plant.get_pan_position()
        g_x,g_y,ob_x,ob_y = 0,0,0,0
        Q = 0.8
        eta = 5
        for ob in self.obs_pos:
            ob_x = ob[0]
            ob_y = ob[1]
            if math.sqrt((bot_x-ob_x)*(bot_x-ob_x)+(bot_y-ob_y)*(bot_y-ob_y)) < Q:
                # print('get_ob:',ob[0],ob[1],'self_pos:',bot_x,bot_y,' d:',math.sqrt((bot_x-ob_x)*(bot_x-ob_x)+(bot_y-ob_y)*(bot_y-ob_y)))
                d = (bot_x-ob_x)**2+(bot_y-ob_y)**2+delta
                g_x += eta*(1/(math.sqrt(d))-1/Q)*(d)**(-1.5)*(bot_x - ob_x)
                g_y += eta*(1/(math.sqrt(d))-1/Q)*(d)**(-1.5)*(bot_y - ob_y)
                # p.addUserDebugPoints([[ob_x,ob_y,-0.5]],pointColorsRGB=[[1,0,0]],pointSize=10,lifeTime =1)
            else:
                # print('un_get_ob:',ob[0],ob[1],'self_pos:',bot_x,bot_y,'d:',math.sqrt((bot_x-ob_x)*(bot_x-ob_x)+(bot_y-ob_y)*(bot_y-ob_y)))
                pass
        # print('v:',g_x,g_y)
        # print("*******************************")
        _,__,jac2 = self.plant.get_jacob(3,[0,1])
        # g_x,g_y = 0,0 # for test
        task2 = ([g_x,g_y],jac2)
        # print('task2:',task2)
        # print('g:',(g_x,g_y))


        # joint position bounds
        bot_id = self.plant.pandaUid
        q_dot = []
        # q1 = p.getJointState(bot_id, bound_joints[0])[0]
        # q2 = p.getJointState(bot_id, bound_joints[1])[0]
        # q3 = p.getJointState(bot_id, bound_joints[2])[0]
        # q1_max = p.getJointInfo(bot_id, bound_joints[2])[8]
        # q1_min = p.getJointInfo(bot_id, bound_joints[2])[9]
        # print('q1_min',q1_min)
        # print('q1_max',q1_max)
        # print('q1',q1)
        delta = 0.1
        pp = 1
        for i,index in enumerate(self.bound_joints):
            q = p.getJointState(bot_id, index)[0]
            max = self.bound[i][1]
            min = self.bound[i][0]
            d = 0
            # print('q',q)
            if q > self.bound[i][3]:
                d = max - q
                # Q = max - bound[i][3]
                # q_dot.append(1/Q - 1/d)
                q_dot.append(pp*-1/(d*d + delta))
            elif q < self.bound[i][2]:
                d = q - min
                # Q = self.bound[i][2] - min
                q_dot.append(pp*1/(d*d + delta))
            else:
                q_dot.append(0)
        # print('joint_position',self.plant.get_joint_position())
        # print('q_dot',q_dot)
        jac3 = np.zeros((5,9))
        jac3[0][4] = 1
        jac3[1][5] = 1
        jac3[2][6] = 1
        jac3[3][7] = 1
        jac3[4][8] = 1
        # q_dot = [0,0,0,0,0] # for test
        task3 = (q_dot,jac3)
        # print('task1',task1)
        # print('task',task3)


        # avoid self collision 
        bank_pos = self.plant.get_single_link_pos(2)+np.array([0,0,0.3])
        link6_pos = self.plant.get_single_link_pos(6)
        link7_pos = self.plant.get_single_link_pos(7)

        # for link7 
        d7 = np.linalg.norm(bank_pos - link7_pos)
        Qd7 = 0.6
        etad7 = 0.3
        gd7_x,gd7_y,gd7_z = 0,0,0
        if d7 < Qd7:
            gd7_x = etad7*(1/d7-1/Qd7)*(d7)**(-3)*(link7_pos[0] - bank_pos[0])
            gd7_y = etad7*(1/d7-1/Qd7)*(d7)**(-3)*(link7_pos[1] - bank_pos[1])
            gd7_z = etad7*(1/d7-1/Qd7)*(d7)**(-3)*(link7_pos[2] - bank_pos[2])
        _,__,jac4 = self.plant.get_jacob(8,[0,2])
        task4 = ([gd7_x,gd7_y,gd7_z],jac4)
        



        return task1,[task2,task3]
        

    def generate_obs(self,num):
        # generate a list of position
        pos_list = []
        scale = 3
        offset = 1
        for i in range(num):
            # x = np.random.rand() - 0.5
            # x = (scale*x + offset) if x > 0 else (scale*x - offset)
            # y = np.random.rand() - 0.5
            # y = (scale*y + offset) if y > 0 else (scale*y - offset)
            # pos_list.append([x,y,-0.30])
            x = scale*(np.random.rand() - 0.5)
            y = scale*(np.random.rand() - 0.5)
            while(1):
                if abs(x) > offset or abs(y) > offset:
                    break
                else:
                    x = scale*(np.random.rand() - 0.5)
                    y = scale*(np.random.rand() - 0.5)
            pos_list.append([x,y,-0.3]) 
        # pos_list = [[0.5,0.5,0.01],[-5,5,0.01],[0.7,-0.5,0.01]]
        # print(pos_list)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # put obs on the positions 
        
        shift = [0, 0, 0]
        meshScale = [0.1, 0.1, 0.1]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)
        for pos in pos_list:
            p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=pos,
                                useMaximalCoordinates=True)
        return pos_list
    
    def generate_obs_new(self,num):
        # TO DO: generate obstacles randomly outside of the trajectory 
        # generate a list of position
        pos_list = []
        scale = 6
        offset = 1
        for i in range(num):
            # x = np.random.rand() - 0.5
            # x = (scale*x + offset) if x > 0 else (scale*x - offset)
            # y = np.random.rand() - 0.5
            # y = (scale*y + offset) if y > 0 else (scale*y - offset)
            # pos_list.append([x,y,-0.30])
            x = scale*(np.random.rand() - 0.5)
            y = scale*(np.random.rand() - 0.5)
            while(1):
                if abs(x) > offset or abs(y) > offset:
                    break
                else:
                    x = scale*(np.random.rand() - 0.5)
                    y = scale*(np.random.rand() - 0.5)
            pos_list.append([x,y,-0.3]) 
        # pos_list = [[0.5,0.5,0.01],[-5,5,0.01],[0.7,-0.5,0.01]]
        # print(pos_list)
        for j,pos in enumerate(pos_list):
            for i in range(int(self.num_points/10)):
                des_p = self.traj_pts[10*i]
                if (des_p[0] - pos[0])**2+(des_p[1] - pos[1])**2 < 0.01:
                    if np.random.rand() > 0.6:
                        # pos_list.remove(j)
                        del pos_list[j]
                        break

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # put obs on the positions 
        
        shift = [0, 0, 0]
        meshScale = [0.1, 0.1, 0.1]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)
        for pos in pos_list:
            p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=pos,
                                useMaximalCoordinates=True)
        return pos_list
    
    def generate_obs_new_new(self,num):
        step_len = 1.8
        step_num = 6
        # generate a list of position
        pos_list = []
        for i in range(step_num):
            for j in range(step_num):
                x = (i - int(step_num/2))*step_len
                y = (j - int(step_num/2))*step_len
                if math.fabs(x) > 1 or math.fabs(y) > 1:
                    pos_list.append([x,y,-0.3]) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # put obs on the positions 
        
        shift = [0, 0, 0]
        meshScale = [0.1, 0.1, 0.1]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)
        for pos in pos_list:
            p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=pos,
                                useMaximalCoordinates=True)
        return pos_list
        



        
    
