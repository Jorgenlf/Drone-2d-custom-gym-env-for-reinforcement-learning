from stable_baselines3 import PPO
import gym
import time
import sys

from drone_2d_env import * 

from gym.envs.registration import register

register(
    id='drone-2d-custom-v0',
    entry_point='drone_2d_env:Drone2dEnv',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': False,
            'initial_throw': True}
)

continuous_mode = True #if True, after completing one episode the next one will start automatically
random_action = False #if True, the agent will take actions randomly

render_sim = True #if True, a graphic is generated

env = gym.make('drone-2d-custom-v0', render_sim=render_sim, render_path=True, render_shade=True,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True, initial_throw=True)


model = PPO.load("./ppo_agents/ppo_agent.zip" ,env) #ppo_agent.zip works nicely

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs = env.reset()

try:
    while True:
        if render_sim:
            env.render()

        if random_action:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        if done is True:
            if continuous_mode is True:
                state = env.reset()
            else:
                break

finally:
    env.close()
