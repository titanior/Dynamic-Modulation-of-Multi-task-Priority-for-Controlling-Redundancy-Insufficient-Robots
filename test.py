from MMP.my_env import my_env
import numpy as np
env = my_env()
env.render()
s = [0,1.0,0,0,0,0,1.0,0,0]
for _ in range(10000):
    env.step(s)
env.close()
