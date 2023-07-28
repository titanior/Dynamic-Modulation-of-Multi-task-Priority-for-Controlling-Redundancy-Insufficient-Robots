from MMP.MMP_env import PandaEnv as env
import numpy as np
from numpy.linalg import pinv,inv,svd
import math
import time
# from numpy import dot
class task_modulator():
    def __init__(self):
        # a list of (x_dot,jacobian)
        self.subtasks = []
        self.uniti_tasks = []
        self.S = None
        self.A = None
        self.P = None
        self.J_sub = []
        self.x_S = []
        self.A_dot = []
        self.x1= []
        self.jac1 = []
        self.flag = False
        self.dt = 0
        self.start = 0
        self.end = 0.000001
    
    def update_tasks(self,task1,task2):
        self.x1 = task1[0]
        self.jac1 = task1[1]
        self.subtasks = task2

    def unitization(self):
        self.J_sub = []
        self.x_S = []
        for (task,jacob) in self.subtasks:
            self.uniti_tasks.extend(zip(task,jacob))  
            self.J_sub.extend(jacob) 
            self.x_S.extend(task)
        # print('j_sub:',self.J_sub)
            
    def update_para(self):
        l = len(self.x_S)
        n_m = 9 - len(self.x1)
        gamma = 0.8
        d = 0.1
        k = 50
        dt = 0.5
        # if self.end == 0:
        #     dt = 0
        #     print('iniiiiiiiitialize')
        # else:
        #     dt = 500000*( self.end - self.start)
            # print('dt',dt)
        
        if self.P is None:
            # initialization
            # self.A = np.append(gamma*np.eye(n_m),np.zeros((n_m,(l-n_m))))
            # print(n_m)
            # print(l)
            self.A = np.array(np.concatenate((gamma*np.eye(n_m), np.zeros((n_m,(l-n_m)))), axis=1))
            # print(11111111111)
            # print(self.A)
            # print(gamma*np.eye(n_m))
            # print(np.zeros((n_m,(l-n_m))))
            self.S = np.zeros((l,l))
            self.P = np.zeros((n_m,l))
            self.A_dot = np.zeros((n_m,l))
            
        else:
            # update S
            # i = 0
            
            # print('x_s',self.x_S)
            # for i,task in enumerate(self.x_S):
            #     self.S[i][i] = 1.0/(1+math.exp(k*(d+task)))+1.0/(1+math.exp(k*(d-task)))
                # print(1/(1+math.exp(k*(d+0)))+1/(1+math.exp(k*(d-0))))
            # print('S:',self.S)
            # update P
            for i,p_i in enumerate(self.P):
                for j,p_ij in enumerate(p_i):
                    p_ij = 1
                    for u in range(i):
                        p_ij *= (1-self.A[u][j])
                    for v in range(j):
                        p_ij *= (1-self.A[i][v])
                    for u in range(n_m):
                        if i != u:
                             p_ij *= (gamma-self.A[u][j])
                    self.P[i][j] = p_ij
            # print('P:',self.P)
            # update A
            self.A_dot = np.dot(self.P,self.S)
            # print('A_dot:','\n',self.A_dot)
            for i,Ad_i in enumerate(self.A_dot):
                w = np.argmax(Ad_i)
                if self.A[i][w] == gamma: 
                    self.A_dot[i] = 0
                else:
                    # v = np.argmax((Ad_i-self.A_dot[i][w]))
                    v = np.argsort(Ad_i)[-2]
                    z = (self.A_dot[i][w]+self.A_dot[i][v])/2
                    self.A_dot[i] = self.A_dot[i]-z
                    s = 0
                    for j in range(l):
                        if self.A_dot[i][j] > 0 or self.A[i][j] != 0:
                            s = s + self.A_dot[i][j]
                    self.A_dot[i][w] = self.A_dot[i][w]-s
            # print('2.0 A_dot:\n',self.A_dot)
            # self.A = self.A + self.A_dot*dt
            for i,Ad_i in enumerate(self.A_dot):
                for j,Ad_ij in enumerate(Ad_i):
                    A_ij = gamma if gamma < (self.A[i][j]+Ad_ij*dt) else (self.A[i][j]+Ad_ij*dt)
                    A_ij = A_ij if A_ij > 0 else 0
                    self.A[i][j] = A_ij
                self.A[i] = gamma*self.A[i]/self.A[i].sum()

                        

    def merge(self,env):
        # self.start = time.time()
        # update tasks
        # self.update_tasks()
        # _,__,jac2 = env.get_jacob(3,[0,2])
        # _,__,jac3 = env.get_jacob(3,[3,5])
        # # print('jac2',jac4)
        # _,__,jac4 = env.get_jacob(11,[3,5])
        # _,__,jac1 = env.get_jacob(11,[0,2])
        # x1 = [0,0,0]
        # x2 = [0,0.2,0]
        # x4 = [0.5,0,0]
        # x3 = [0,0,0]
        # # self.subtasks = [(x2,jac2),(x3,jac3)]
        # # self.subtasks = [(x2,jac2)]
        # self.update_tasks((x1,jac1),[(x2,jac2),(x3,jac3),(x4,jac4)])
        self.unitization()

        # update A,S,P
        self.update_para()
        # print(self.A)
        # print(len(jac3))
        # print('jac_sub:',self.J_sub)
        J_sub = np.array(self.J_sub)
        x_S = np.array(self.x_S)
        # merge 
        # print('x_2:',self.dot([self.A,x_S]))
        N1 = np.eye(J_sub.shape[1]) - self.dot([pinv(self.jac1),self.jac1])
        # print('N1:',N1)
        # print('J1*N1:',self.dot([self.jac1,N1]))
        # print('A:\n',self.A)
        # print('x_s:',x_S)
        # print('J_sub:',J_sub.shape)
        # print(self.dot([self.A,J_sub,N1,J_sub.T,self.A.T]))
        v,s1,u  = svd(self.jac1)
        # print('svd task1 s:',s1)
        self.dot([pinv(self.jac1),self.x1])
        ve = self.dot([pinv(self.jac1),self.x1])+self.dot([
            N1,J_sub.T,self.A.T,
            self.dls(self.dot([self.A,J_sub,N1,J_sub.T,self.A.T])),
            # N1,
            # pinv(self.dot([self.A,J_sub,N1])),
            (self.dot([self.A,x_S])-self.dot([self.A,J_sub,pinv(self.jac1),self.x1]))
            # (self.dot([self.A,x_S]))
        ])
        # print_debug =  self.dot([
        #     J_sub.T,self.A.T,
        #     self.dls(self.dot([self.A,J_sub,N1,J_sub.T,self.A.T])),
        #     # N1,
        #     # pinv(self.dot([self.A,J_sub,N1])),
        #     (self.dot([self.A,x_S])-self.dot([self.A,J_sub,pinv(self.jac1),self.x1]))
        #     # (self.dot([self.A,x_S]))
        # ])
        # print('print_debug:',print_debug)
        # print_debug2 = self.dls(self.dot([self.A,J_sub,N1,J_sub.T,self.A.T]))
        # print('print_debug2:',print_debug2)
        # print_debug3 = self.dot([self.A,J_sub,N1,J_sub.T,self.A.T])
        # print('print_debug3:',print_debug3)
        # ve = self.dot([pinv(self.jac1),self.x1])+self.dot([
        #     N1,np.array([-10,1,1,1,1,2,1,1,1])
        #     # (self.dot([self.A,x_S]))
        # ])
        # print("pinv1:",pinv(self.dot([self.A,J_sub,N1])))
        # print("pinv2:",pinv(self.dot([self.A,J_sub,N1])))

        # self.end = time.time()
        return ve
        
    def dot(self,x):
        if len(x) == 2:
            return np.dot(x[0],x[1])
        else:
            return np.dot(x[0],self.dot(x[1:]))

    def get_task(self,env):
        task1 = [1,1,1,0,0,0]
        _,__,jac1 = env.get_jacob(11)
        # print(jac1)
        task2 = [1,0,0,0,0,0]
        _,__,jac2 = env.get_jacob(4)
        tmp = (np.eye(9)-np.dot(pinv(jac1),jac1))
        ve = list(np.dot(pinv(jac1),task1)+np.dot(np.dot(tmp,pinv(np.dot(jac2,tmp))),task2-np.dot(np.dot(jac2,pinv(jac1)),task1)))  


    def dls(self,A):
        # svd
        threshold = 0.008
        # print(A)
        u,s,v = svd(A)
        lam = 0.0000001
        S = np.zeros([s.shape[0],s.shape[0]])
        # return self.dot([A.T,inv(self.dot([A,A.T])+lam*np.eye(A.shape[0]))])
        for i,s_i in enumerate(s):
            S[i][i] = s_i
            if (math.fabs(s_i)<threshold):
                S[i][i] = np.sign(s_i)*threshold
            
        A = self.dot([u,S,v])
        # print(A)
        # return inv(A)
        if(np.linalg.det(A)!=0):
            # print("not singular")
            return inv(A) 
        else:
            # print("singular")
            return self.dot([A.T,inv(self.dot([A,A.T])+lam*np.eye(A.shape[0]))])
    
    def update_s(self,s):
        self.S = np.diag(s)


    def test(self,env):
        # self.subtasks = [([1,2,3],[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3,3]]),([4,5,6],[[4,4,4,4,4],[5,5,5,5,5],[6,6,6,6,6]])]
        # self.unitization()
        # print(self.J_sub)
        print(111111111111111111111111)
        _,__,jac1 = env.get_jacob(11,[0,3])
        
        print('AAAAAAAAAAAAAAAAAAAAA',jac1)

    def test_null_space(self,env,task1,task2):
        # task1 = [0,0,1,0,0,0]
        _,__,jac1 = env.get_jacob(11,[0,6])
        # task2 = [1,0,0,0,0,0]
        _,__,jac2 = env.get_jacob(4,[0,2])
        tmp = (np.eye(9)-np.dot(pinv(jac1),jac1))
        ve = list(np.dot(pinv(jac1),task1)+np.dot(np.dot(tmp,pinv(np.dot(jac2,tmp))),task2-np.dot(np.dot(jac2,pinv(jac1)),task1)))  
        # print(ve)
        return ve
    
    def test_dot(self):
        a = np.array([[1,2],[3,4]])
        b = np.array([[5,6],[7,8]])
        c = np.array([[9,10],[11,12]])
        print(self.dot([a,b,c]))
        print(np.dot(a,np.dot(b,c)))
