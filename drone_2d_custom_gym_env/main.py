from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from PIL import Image
import gym
import time
from multiprocessing import freeze_support, get_context
import multiprocessing
import json

from tensorboardlogger import *
from drone_2d_env import *
from rl_config import rl_config, env_test_config, env_train_config

from gym.envs.registration import register

def red_blue_grad(float):
    '''takes in a float between 0 and 1 and returns a rgb value between red and blue'''
    r = 0
    g = 0
    b = 0

    if float < 0.5:
        r = 255
        b = 255*float*2 
    else:
        r = 255*(1-float)*2
        b = 255
    return (r,g,b)

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
                    if event.key == pygame.K_s:
                        print("Saving screenshot")
                        pygame.image.save(env.screen, "screenshots/screenshot.png")
                        image = Image.open("screenshots/screenshot.png")
                        base_name = "screenshots/pdfs/img_"
                        index = 1
                        pdf_name = f"{base_name}{index}.pdf"
                        while os.path.exists(pdf_name):
                            index += 1
                            pdf_name = f"{base_name}{str(index)}.pdf"
                        image.save(pdf_name, format="PDF")
                    
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

# mode = "train"
single_threaded = False #When false, multithreading used uses all but 2 cores

# mode = "eval"
agent_path = 'ppo_agents/PFCA_see_3_obs_20_90.zip'
# scenarios = ['stage_1','stage_2','stage_3','stage_4','stage_5']
# scenarios = ['parallel','S_parallel','perpendicular','corridor','S_corridor','large','impossible']
# for scenario in scenarios:
    # env_test_config['scenario'] = scenario
mode = "test"
run_n_times = 10
runs = 0
flight_paths = []
apes = []
collisions = []
successes = 0
fails = 0
time_spent = []
rewards = []


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
elif mode == "test":
    #Want to run one of the test scenarios n times and save the results
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
        while runs < run_n_times:
            env.render()

            action, _states = model.predict(obs)

            obs, reward, done, info = env.step(action)

            if done is True:
                if info['n_successful_runs'] == 1:
                    successes += 1
                if info['n_failed_runs'] == 1:
                    fails += 1
                collisions.append(info['n_collisions'])
                flight_paths.append(info['flight_path'])
                apes.append(info['APE'])
                time_spent.append(info['env_steps'])
                rewards.append(info['total_reward'])
                runs += 1
                if continuous_mode is True:
                    state = env.reset()
                else:
                    break
    finally:
        env.close()
        # Saving test results 
        scenario = env_test_config['scenario']
        agent_nr = agent_path.split('_')[-2].split('.')[0]
        file_path = 'Tests/'+scenario+'/test_'+str(len(os.listdir('Tests/'+scenario)))
        os.makedirs(file_path,exist_ok=True)
        time_spent = np.array(time_spent)

        with open(file_path+'/flight_paths', 'w') as json_file:
            json.dump(flight_paths, json_file)

        apes = np.array(apes)
        collisions = np.array(collisions)
        rewards = np.array(rewards)
        time_spent = np.array(time_spent)
        collision_sum = np.sum(collisions)
        np.save(file_path+'/collisions.npy',collisions)    
        np.save(file_path+'/rewards.npy',rewards)
        np.save(file_path+'/apes.npy',apes)
        np.save(file_path+'/time_spent.npy',time_spent)
        with open(file_path+'/'+scenario+'_'+str(agent_nr)+'_results.txt', 'w') as file:
            file.write('Successes: '+str(successes)+'\n')
            file.write('Fails: '+str(fails)+'\n')
            file.write('Collisions: '+str(collision_sum)+'\n')
            file.write('Success rate: '+str(successes/(successes+fails))+'\n')
            file.write('Collision rate: '+str(collision_sum/(successes+fails))+'\n')
            file.write('Average APE: '+str(np.mean(apes))+'\n')
            file.write('Average flight time: '+str(np.mean(time_spent))+'\n')
            file.write('Agent path: '+agent_path+'\n')
        
        #Render all flight paths in a single plot
        obstacles = []
        space = pymunk.Space()
        pymunk.pygame_util.positive_y_is_up = True

        screen_width = env_test_config['screensize_x']
        screen_height = env_test_config['screensize_y']

        if scenario == 'perpendicular':
            wps,predef_path,obstacles=create_test_scenario(space,'perpendicular',screen_width,screen_height)
        if scenario == 'parallel':
            wps,predef_path,obstacles=create_test_scenario(space,'parallel',screen_width,screen_height)
        if scenario == 'S_parallel':
            wps,predef_path,obstacles=create_test_scenario(space,'S_parallel',screen_width,screen_height)
        if scenario == 'corridor':
            wps,predef_path,obstacles=create_test_scenario(space,'corridor',screen_width,screen_height)
        if scenario == 'S_corridor':
            wps,predef_path,obstacles=create_test_scenario(space,'S_corridor',screen_width,screen_height)
        if scenario == 'large':
            wps,predef_path,obstacles=create_test_scenario(space,'large',screen_width,screen_height)
        if scenario == 'impossible':
            wps,predef_path,obstacles=create_test_scenario(space,'impossible',screen_width,screen_height)
        if scenario == 'stage_1' or scenario == 'stage_2' or scenario == 'stage_3' or scenario == 'stage_4' or scenario == 'stage_5':
            wps,predef_path,obstacles=None,None,None
        
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Drone2d Environment")
        screen.fill((243, 243, 243))
        if env_test_config['mode'] == 'test':
            #Draw first wp:
            pygame.draw.circle(screen, (0, 0, 0), (wps[0][0], screen_height-wps[0][1]), 5)
            #Draw final wp:
            pygame.draw.circle(screen, (0, 0, 0), (wps[-1][0], screen_height-wps[-1][1]), 5)

            #Drawing predefined path
            predef_path_coords = predef_path.get_path_coord()
            predef_path_coords = [(x, screen_height-y) for x, y in predef_path_coords]
            pygame.draw.aalines(screen, (0, 0, 0), False, predef_path_coords)

            #Draw obstacles:
            draw_options = pymunk.pygame_util.DrawOptions(screen)
            draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            space.debug_draw(draw_options)

        min_rew = np.min(rewards)
        max_rew = np.max(rewards)
        normd_rews = (rewards-min_rew)/(max_rew-min_rew)
        for i, path in enumerate(flight_paths):
            if len(path) > 2: #Some paths may not be drawn if the drone crashes immediately
                color = red_blue_grad(normd_rews[i])
                if collisions[i] == 1:
                    pygame.draw.aalines(screen, (255, 0, 0), False, path, 1)
                else:
                    pygame.draw.aalines(screen, color, False, path, 1)
        else: pass

        #Draw a color bar explaining the color coding of the flight paths blue = high reward, red = low reward
        for i in range(100):
            pygame.draw.line(screen, red_blue_grad(i/100), (screen_width-100, screen_height-900-i), (screen_width-50, screen_height-900-i), 1)

        font = pygame.font.SysFont('Arial', 30)
        text = font.render('High reward', True, (0,0,0))
        screen.blit(text, (screen_width-140, screen_height-1030))

        font = pygame.font.SysFont('Arial', 30)
        text = font.render('Low reward', True, (0,0,0))
        screen.blit(text, (screen_width-140, screen_height-910))

        pygame.display.flip()
        pygame.image.save(screen, file_path+'/'+scenario+'_'+str(agent_nr)+'.png')