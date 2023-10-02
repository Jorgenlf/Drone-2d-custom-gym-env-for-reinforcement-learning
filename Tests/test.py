import gym

from simple_test_env import *
from gym.envs.registration import register

##"Init"
register(
    id='TestEnv-v0',
    entry_point='simple_test_env:testEnv',
    kwargs={'n_steps': 500,}
)
##

env = gym.make('TestEnv-v0', n_steps=500)

env.reset()

continuous_mode = True

try:
    while True:
        env.render()

        obs,reward,done,info = env.step(True)

        if done is True:
            if continuous_mode is True:
                state = env.reset()
            else:
                break

finally:
    env.close()