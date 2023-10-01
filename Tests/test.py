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

try:
    while True:
        env.render()

finally:
    env.close()