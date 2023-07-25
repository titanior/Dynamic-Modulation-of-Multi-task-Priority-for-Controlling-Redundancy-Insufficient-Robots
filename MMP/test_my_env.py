from my_env import my_env
import numpy as np
env = my_env()
env.render()
env.reset()
s = [1,1,1,0,0,0,0]
for _ in range(10000):
    env.step(s)
    # env.plant.draw_pan()
env.close()
