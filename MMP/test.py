import gym
import MMP_env as MMP_env
from my_env import my_env as my_env
import numpy as np
from task_modulate import task_modulator
import time
from traj_generator import Bezier,point_generator
import pybullet as p
# env = MMP_env.PandaEnv()
env = my_env()
env.reset()
m = task_modulator()
task1 = [0,0,0,0,0,0]
task2 = [0,1,0] 
count =0
start = 0
end = 0
x1 = [0,0,0]
x2 = [0,0.2,0]
x4 = [1.5,0,0]
x3 = [0,0,0]
points=(np.array([
        [1,3,0],
        [1.5,1,0],
        [4,2,0],
        [4,3,3],
        [2,3,3],
        [5,5,2],
        [9,7,4],
        [12,4,-3]
        ])-np.array([1,3,0]))*0.2
pt_gen = point_generator()
points = pt_gen.get_points(10)
bz=Bezier(points,2000)
v = bz.get_v(0,5)
ps = bz.getBezierPoints(0)
env.render()
p.setRealTimeSimulation(0)  # 关闭实时模拟
p.setTimeStep(1./240.)  # 设置时间步
# p.addUserDebugLine(self.last_link_trn,link_trn,lineColorRGB=[0,0,1])
start = time.time()
result = p.getLinkState(env.pandaUid,
                        env.ur5EndEffectorIndex,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
offset = link_trn
for _ in range(1900):
    m.start = time.time()
    count += 1
    i_v = int(2000*(count/240)/5)
    p.addUserDebugLine(offset+ps[i_v+1],offset+ps[i_v],lineColorRGB=[0,1,0])
    env.draw_end_eff()
    # if (count>50):
    #     # count = 0
    #     # task1[0] = -task1[0]
    #     task2[1] -= 1/50
    # if (count>50):
    #     # count = 0
    #     # task1[0] = -task1[0]
    #     task2[1] -= 1/50
    
    # print(1)
    # env.step(np.array([0, 0, 0 , 0 ,0  ,0 ,0, 0 , 1])) # take a random action
    # env.get_jacob(4)
    # env.print_joint_info(2)
    # env.test()
    # ve = m.test_null_space(env,task1,task2)
    x1 = v[i_v]
    if(count>100):
        x2 = [0.3,0,0]
    if(count>200):
        x4 = [0,2,0]
        x3 = [0,0,3]
    _,__,jac2 = env.get_jacob(3,[0,2])
    _,__,jac3 = env.get_jacob(3,[3,5])
    # print('jac2',jac4)
    _,__,jac4 = env.get_jacob(11,[3,5])
    _,__,jac1 = env.get_jacob(11,[0,2])

    # self.subtasks = [(x2,jac2),(x3,jac3)]
    # self.subtasks = [(x2,jac2)]
    m.update_tasks((x1,jac1),[(x1,jac1),(x1,jac1),(x1,jac1)])
    ve = m.merge(env)
    print('ve',ve)
    env.step(ve)
    m.end = time.time()
end = time.time()
print('time:',end-start)
while 1:
    pass
env.close()


