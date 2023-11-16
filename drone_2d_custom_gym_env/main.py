from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import gym
import time
from multiprocessing import freeze_support, get_context
import multiprocessing

from tensorboardlogger import *
from drone_2d_env import * 
from rl_config import rl_config, env_test_config, env_train_config

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

#---------------------------------#
total_timesteps = rl_config['total_timesteps']

# mode = 'debug'

mode = "train"
single_threaded = False #When false, multithreading used uses all but 2 cores

# mode = "eval"
agent_path = 'ppo_agents/PFCA_see_3_obs_8_40.zip' 

#PFCA_20 is PFCA 4 on homecomputer 
#PFCA_21 is PFCA 5 on homecomputer using all 4 init positions of path better result!
#PFCA_22 is PFCA 6 on homecomputer using all 4 init positions and NEW CA reward function
#PFCA_23 is PFCA 7 on homecomputer using all 4 init positions and NEW CA reward function and doubeled PA reward i.e. [-2,2] rather than [-1,1] follows tighter but crashes more often
#PFCA_24 is PFCA 8 on homecomputer using all 4 init positions and NEW CA reward function and doubled PA reward i.e. [-2,2] rather than [-1,1] uses the lambda_CA and lambda_PA hyperparameters. spoiled by obstacle spawning whole training
#PFCA 25 is PFCA 9 --||-- but correct curriculum learning
#PFCA 26 is PFCA 10 --||-- curriculum learning with random obsspawn after 2M timesteps
#PFCA_see_3_obs_1_34.zip sees 3 obstacles performs fairly well

continuous_mode = True #if True, after completing one episode the next one will start automatically relevant for eval mode
#---------------------------------#

if mode == "debug":
    # Inspect an environment manually
    register(
        id='drone-2d-test',
        entry_point='drone_2d_env:Drone2dEnv',
        kwargs=env_test_config
    )

    env = gym.make('drone-2d-test')
    _manual_control(env)
    exit()

elif mode == "train":
    register(
        id='drone-2d-train',
        entry_point='drone_2d_env:Drone2dEnv',
        kwargs=env_train_config
    )
    env = None
    if single_threaded is True:
        num_cpu = 1
        env = gym.make('drone-2d-train')
        # Init callbacks #TODO make a smart folder structure
        tensorboard_logger = TensorboardLogger()
        checkpoint_saver = CheckpointCallback(save_freq=100000 // num_cpu,
                                                save_path="logs",
                                                name_prefix="rl_model",
                                                verbose=True)
        # List of callbacks to be called
        callbacks = CallbackList([tensorboard_logger, checkpoint_saver])

        model = PPO("MlpPolicy", env, verbose=True,tensorboard_log="logs")

        with open('logs/rl_config.txt', 'w') as file:
            file.write(str(env_train_config))

        with open('logs/rl_config.txt', 'w') as file:
            file.write(str(rl_config))

        model.learn(total_timesteps=total_timesteps,tb_log_name='PPO_PFCA_see_3_obs', callback=callbacks)
        model.save('new_agent')
        env.close()

    else:
        if __name__ == '__main__':
            print('CPU COUNT:', multiprocessing.cpu_count())
            max_cpu = multiprocessing.cpu_count()
            num_cpu = max_cpu-2
            print('Using',num_cpu,'CPUs')

            freeze_support()
            ctx = multiprocessing.get_context('spawn')
            env_id = 'drone-2d-train' 
            env = SubprocVecEnv([make_mp_env(env_id=env_id, rank=i) for i in range(num_cpu)])

            tensorboard_logger = TensorboardLogger()
            checkpoint_saver = CheckpointCallback(save_freq=100000 // num_cpu,
                                                    save_path="logs",
                                                    name_prefix="rl_model",
                                                    verbose=True)
            # List of callbacks to be called
            callbacks = CallbackList([tensorboard_logger, checkpoint_saver])

            model = PPO("MlpPolicy", env, verbose=True,tensorboard_log="logs",ent_coef=0.01)

            with open('logs/env_train_config.txt', 'w') as file:
                file.write(str(env_train_config))
            
            with open('logs/rl_config.txt', 'w') as file:
                file.write(str(rl_config))

            model.learn(total_timesteps=total_timesteps,tb_log_name='PPO_PFCA_see_3_obs', callback=callbacks)
            model.save('new_agent')
            env.close()

elif mode == "eval":
    register(
        id='drone-2d-test',
        entry_point='drone_2d_env:Drone2dEnv',
        kwargs=env_test_config
    )
    env = gym.make('drone-2d-test')

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
