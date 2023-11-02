from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import gym
import time
from multiprocessing import freeze_support, get_context
import multiprocessing

from tensorboardlogger import *
from drone_2d_env import * 

from gym.envs.registration import register

    
def _manual_control(env):
    """ Manual control function.
        Reads keyboard inputs and maps them to valid inputs to the environment.
    """
    # Infinite environment loop:
    # - Map keyboard inputs to valid actions
    # - Reset environment once done is True
    # - Exits upon closing the window or pressing ESCAPE.
    state = env.reset()
    input = [0,0]
    while True:
        env.render()
        try:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        input = [1,-1]
                    if event.key == pygame.K_LEFT:
                        input = [-1,1]
                    if event.key == pygame.K_UP:
                        input = [1,1]   
                    if event.key == pygame.K_DOWN:
                        input = [-1,-1]

                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return

                obs, rew, done, info = env.step(action=input)
                if done:
                    done = False
                    env.reset()

                if event.type == pygame.QUIT:
                    print("Pygame window closed, exiting")
                    env.close()
                    return

        except KeyboardInterrupt:
            print("CTRL-C pressed, exiting.")
            env.close()
            return


def make_mp_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.reset() #seed=seed + rank #TODO should maybe pass this in to the reset fcn...
        return env
    set_random_seed(seed=seed)
    return _init

register(
    id='drone-2d-custom-v0',
    entry_point='drone_2d_env:Drone2dEnv',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_steps': 900, 'n_fall_steps': 5, 'change_target': False,
            'initial_throw': True, 'random_path_spawn':True,
            'path_segment_length':100, 'n_wps': 8,'screensize_x':1000,'screensize_y':1000}
)

#---------------------------------#
# mode = 'debug'

mode = "train"
single_threaded = False #When false, multithreading used

mode = "eval"
agent_path = 'ppo_agents/maxObs_PFonly.zip' 
continuous_mode = True #if True, after completing one episode the next one will start automatically relevant for eval mode
#---------------------------------#

if mode == "debug":
    # Inspect an environment manually
    env = gym.make('drone-2d-custom-v0', render_sim=True, render_path=True, render_shade=True,
            shade_distance=70, n_steps=900, n_fall_steps=0, change_target=True, initial_throw=False)

    _manual_control(env)
    exit()

elif mode == "train":
    env = None
    if single_threaded is True:
        num_cpu = 1
        env = gym.make('drone-2d-custom-v0', render_sim=False, render_path=False, render_shade=False,
                    shade_distance=70, n_steps=900, n_fall_steps=0, change_target=True, initial_throw=False,screensize_x =1000, screensize_y=1000)
        # Init callbacks #TODO make a smart folder structure
        tensorboard_logger = TensorboardLogger()
        checkpoint_saver = CheckpointCallback(save_freq=100000 // num_cpu,
                                                save_path="logs",
                                                name_prefix="rl_model",
                                                verbose=True)
        # List of callbacks to be called
        callbacks = CallbackList([tensorboard_logger, checkpoint_saver])

        model = PPO("MlpPolicy", env, verbose=True,tensorboard_log="logs")

        model.learn(total_timesteps=1800000,tb_log_name='PPO_tb_log', callback=callbacks)
        model.save('new_agent')
        env.close()

    else:
        if __name__ == '__main__':
            print('CPU COUNT:', multiprocessing.cpu_count())
            max_cpu = multiprocessing.cpu_count()
            num_cpu = max_cpu-2

            freeze_support()
            ctx = multiprocessing.get_context('spawn')
            env_id = 'drone-2d-custom-v0' 
            env = SubprocVecEnv([make_mp_env(env_id=env_id, rank=i) for i in range(num_cpu)])

            # Init callbacks #TODO make a smart folder structure
            tensorboard_logger = TensorboardLogger()
            checkpoint_saver = CheckpointCallback(save_freq=100000 // num_cpu,
                                                    save_path="logs",
                                                    name_prefix="rl_model",
                                                    verbose=True)
            # List of callbacks to be called
            callbacks = CallbackList([tensorboard_logger, checkpoint_saver])

            model = PPO("MlpPolicy", env, verbose=True,tensorboard_log="logs")

            model.learn(total_timesteps=1800000,tb_log_name='PPO_PF', callback=callbacks)
            model.save('new_agent')
            env.close()

elif mode == "eval":
    env = gym.make('drone-2d-custom-v0', render_sim=True, render_path=True, render_shade=True,
                shade_distance=70, n_steps=900, n_fall_steps=5, change_target=True, initial_throw=True)

    model = PPO.load(agent_path,env)

    random_seed = int(time.time())
    model.set_random_seed(random_seed)

    obs = env.reset()

    try:
        while True:
            env.render()

            action, _states = model.predict(obs)

            obs, reward, done, info = env.step(action)

            if done is True:
                if continuous_mode is True:
                    state = env.reset()
                else:
                    break
    finally:
        env.close()
else: 
    print("Invalid mode\nValid modes are: debug, train, eval")
    print("""
⢀⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ 
⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠁⠸⣼⡿ 
⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⠀⠀⠀⠀ 
⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠿⠿⠿⠛⠉""") #To make it obvious that the input is invalid may remove later
