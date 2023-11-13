from Drone import *
from obstacles import *
from event_handler import *
from predef_path import *
from transformations import *

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import os
import glob

#Might make into own file such as eventhandler.py
def drone_obstacle_collision(arbiter, space, data):
    space.collison = True
    return True


class Drone2dEnv(gym.Env):
    """
    render_sim: (bool) if true, a graphic is generated
    render_path: (bool) if true, the drone's path is drawn
    render_shade: (bool) if true, the drone's shade is drawn
    shade_distance: (int) distance between consecutive drone's shades
    n_steps: (int) number of time steps
    n_fall_steps: (int) the number of initial steps for which the drone can't do anything
    change_target: (bool) if true, mouse click change target positions
    initial_throw: (bool) if true, the drone is initially thrown with random force
    """
    def __init__(self, **kwargs):
        render_sim = kwargs['render_sim']
        render_path = kwargs['render_path']
        render_shade = kwargs['render_shade']
        shade_distance = kwargs['shade_distance']
        n_steps = kwargs['n_steps']
        n_fall_steps = kwargs['n_fall_steps']
        change_target = kwargs['change_target']
        initial_throw = kwargs['initial_throw']
        random_path_spawn = kwargs['random_path_spawn']
        path_segment_length = kwargs['path_segment_length']
        n_wps = kwargs['n_wps']
        screensize_x = kwargs['screensize_x']
        screensize_y = kwargs['screensize_y']
        self.lookahead = kwargs['lookahead']
        self.spawn_corners = kwargs['spawn_corners']
        self.danger_range = kwargs['danger_range']
        self.danger_angle = kwargs['danger_angle']
        self.abs_inv_CA_min_rew = kwargs['abs_inv_CA_min_rew']
        self.PA_band_edge = kwargs['PA_band_edge']
        self.PA_scale = kwargs['PA_scale']
        self.PP_vel_scale = kwargs['PP_vel_scale']
        self.PP_rew_max = kwargs['PP_rew_max']
        self.PP_rew_min = kwargs['PP_rew_min']
        self.rew_collision = kwargs['rew_collision']
        self.reach_end_radius = kwargs['reach_end_radius']
        self.rew_reach_end = kwargs['rew_reach_end']
        self.AA_angle = kwargs['AA_angle']
        self.AA_band = kwargs['AA_band']
        self.rew_AA = kwargs['rew_AA']
        self.k_obs = 3 #Number of obstacles the drone has angle and distance to must write more code to make this dynamic

        self.kwargs = kwargs

        #To do curriculum learning
        #Jank way to get timestep by looking at the name of the saved model
        #NB requires the model to be saved as rl_model_*.zip
        #Requires the user to move the zips before training a new model
        # files = glob.glob('C:\CodeThatsSusceptibleToOneDrive\Specialization project\Drone-2d-custom-gym-env-for-reinforcement-learning\logs/rl_model_*.zip')
        files = glob.glob('logs/rl_model_*.zip')
        numbers = [int(file.split('_')[-2]) for file in files]
        self.sim_num = 0
        if numbers == []:
            self.sim_num = 0
        else:
            self.sim_num = max(numbers)
            # print("timestep:", self.sim_num)

        #Rendering booleans
        self.render_sim = render_sim
        self.render_path = render_path
        self.render_shade = render_shade
        
        #Predefined path generation
        self.random_path_spawn = random_path_spawn
        self.wps = []
        self.seg_length = path_segment_length
        if random_path_spawn is True:
            spawn = random.randint(self.spawn_corners[0],self.spawn_corners[1]) 
            scen = ''
            if spawn == 1: #TODO make enum rather, but this works temporarily
                scen = 'DL'
            elif spawn == 2:
                scen = 'DR'
            elif spawn == 3:
                scen = 'UL'
            elif spawn == 4:
                scen = 'UR'

            self.wps = generate_random_waypoints_2d(n_wps,self.seg_length,scen,screen_x=screensize_x,screen_y=screensize_y)
        else:
            self.wps = generate_random_waypoints_2d(n_wps,self.seg_length,'DR',screen_x=screensize_x,screen_y=screensize_y)
        self.predef_path = QPMI2D(self.wps)
        self.waypoint_index = 0

        #Make last waypoint the target
        self.x_target = self.wps[-1][0]
        self.y_target = self.wps[-1][1]

        #Rendering initialization
        self.screen_width = screensize_x
        self.screen_height = screensize_y
        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []
            self.drop_path = []
            self.path_drone_shade = []
            self.draw_red_velocity = False
            self.draw_orange_obst_vec = False
            self.obs_angle = 0
            self.vel_angle = 0
            self.closest_point = np.array([0,0])
            self.drone_alpha = 0
            self.drone_vel = np.array([0,0])

        #Initial values
        self.first_step = True
        self.LA_in_last_wp = False
        self.done = False
        self.info = {}
        self.info['reward'] = 0
        self.info['collision_avoidance_reward'] = 0
        self.info['path_adherence'] = 0
        self.info["path_progression"] = 0
        self.info['collision_reward'] = 0
        self.info['reach_end_reward'] = 0
        self.info['agressive_alpha_reward'] = 0

        self.current_time_step = 0
        self.left_force = -1
        self.right_force = -1
        
        #Pymunk initialization
        self.obstacles = []
        self.init_pymunk()

        #Parameters
        self.max_time_steps = n_steps
        self.stabilisation_delay = n_fall_steps
        self.drone_shade_distance = shade_distance
        self.force_scale = 1000
        self.initial_throw = initial_throw
        self.change_target = change_target

        #Action and observation space
        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        #Max obs config 
        min_observation = np.full(25, -1, dtype=np.float32)
        max_observation = np.full(25, 1, dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)

        #Debugging
        # self.vec = self.debug_path_angle()

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Drone2d Environment")
        self.clock = pygame.time.Clock()

        script_dir = os.path.dirname(__file__)
        icon_path = os.path.join("img", "icon.png")
        icon_path = os.path.join(script_dir, icon_path)
        pygame.display.set_icon(pygame.image.load(icon_path))

        img_path = os.path.join("img", "shade.png")
        img_path = os.path.join(script_dir, img_path)
        self.shade_image = pygame.image.load(img_path)

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000)

        self.space.collison = False

        # Register collision handler
        drone_obstacle_handler = self.space.add_collision_handler(1, 2) #Drone is collision type 1, obstacle is collision type 2
        drone_obstacle_handler.begin = drone_obstacle_collision

        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True

        #Generating drone's starting position
        # random_x = random.uniform(100, 200)
        # random_y = random.uniform(100, 200)
        x1 = self.wps[0][0]
        y1 = self.wps[0][1]

        angle_rand = random.uniform(-np.pi/4, np.pi/4)
        self.drone = Drone(x1, y1, angle_rand, 20, 100, 0.2, 0.4, 0.4, self.space)

        self.drone_radius = self.drone.drone_radius #=20/2 + 100/2 = 60

        damping_factor = 0.9
        #Amount of simple damping to apply to the space.
        # A value of 0.9 means that each body will lose 10% of its velocity per second. Defaults to 1. Like gravity, it can be overridden on a per body basis.
        for body in self.space.bodies:
            body.damping = damping_factor

        #Curriculum 
        if self.sim_num < 800000:
            self.obstacles = []
        elif self.sim_num > 800000 and self.sim_num < 2000000:            
            tmin = 800000
            tmax = 2000000
            cmin = 0.2
            cmax = 1
            spawn = (self.sim_num - tmin)*(cmax-cmin)/(tmax-tmin) + cmin
            spawn = np.random.binomial(1,spawn)
            if spawn == 1: 
                self.obstacles = generate_obstacles_around_path(1, self.space, self.predef_path, 0, 0,onPath=True)
        else:
            n_obs = np.random.normal(1, 4)
            if n_obs < 0: n_obs = 2
            self.obstacles = generate_obstacles_around_path(n_obs, self.space, self.predef_path, 0, 100)

        if self.k_obs > len(self.obstacles):
            self.k_obs = len(self.obstacles)

        if self.render_sim is True:
            self.drone_pos = np.array([x1,y1])
            self.look_ahead_point = self.predef_path.get_lookahead_point(self.drone_pos, self.lookahead)
            LA_body = np.matmul(R_w_b(self.drone.frame_shape.body.angle), self.look_ahead_point - self.drone_pos)
            look_ahead_angle_b_hori = np.arctan2(LA_body[1],LA_body[0]) #range -pi to pi
            look_ahead_angle = ssa(look_ahead_angle_b_hori - self.drone.frame_shape.body.angle)
            if look_ahead_angle < 0: look_ahead_angle += 2*np.pi
            self.look_ahead_angle = look_ahead_angle
        
    def step(self, action):
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_shade is True: self.add_drone_shade()


        self.left_force = (action[0]/2 + 0.5) * self.force_scale
        self.right_force = (action[1]/2 + 0.5) * self.force_scale

        self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.left_force), (-self.drone_radius, 0))
        self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.right_force), (self.drone_radius, 0))

        self.space.step(1.0/60)
        self.current_time_step += 1

        #Saving drone's position for drawing
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()
            self.first_step = False
        else:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()
        if self.render_sim is True and self.render_shade is True:
            x, y = self.drone.frame_shape.body.position
            if np.abs(self.shade_x-x) > self.drone_shade_distance or np.abs(self.shade_y-y) > self.drone_shade_distance:
                self.add_drone_shade()

        #Taking in observations
        obs = self.get_observation()
        drone_vel_x = self.invm1to1(obs[0],-1330,1330)
        drone_vel_y = self.invm1to1(obs[1],-1330,1330)
        drone_omega = obs[2]
        drone_alpha = (obs[3])*np.pi #Does not use sin and cos as alpha angle will not exceed +-90 degrees and the jump in obs is at +-180 degrees
        #drone_alpha is in range -pi to pi
        target_dist_x = self.invm1to1(obs[4],0,self.screen_width)
        target_dist_y = self.invm1to1(obs[5],0,self.screen_height)
        drone_pos_x = self.invm1to1(obs[6],0,self.screen_width)
        drone_pos_y = self.invm1to1(obs[7],0,self.screen_height)
        # 8, 9, 10, 11 and 12 part of collision avoidance so only used if there are obstacles
        s_drone_vel_angle = obs[17]*np.pi
        c_drone_vel_angle = obs[18]*np.pi
        drone_vel_angle = (np.arctan2(s_drone_vel_angle, c_drone_vel_angle) + 2*np.pi)%(2*np.pi) #range 0 to 2pi
        closest_point_x = self.invm1to1(obs[19], 0, self.screen_width)
        closest_point_y = self.invm1to1(obs[20], 0, self.screen_height)
        look_ahead_x = self.invm1to1(obs[21], 0, self.screen_width)
        look_ahead_y = self.invm1to1(obs[22], 0, self.screen_height)
        s_look_ahead_angle = obs[23]
        c_look_ahead_angle = obs[24]
        look_ahead_angle = (np.arctan2(s_look_ahead_angle, c_look_ahead_angle) + 2*np.pi)%(2*np.pi) #range 0 to 2pi
    
        if self.render_sim is True:
            self.vel_angle = drone_vel_angle
            self.drone_pos = np.array([drone_pos_x, drone_pos_y])
            self.drone_alpha = drone_alpha
            self.drone_vel = np.array([drone_vel_x, drone_vel_y])
            self.closest_point = np.array([closest_point_x, closest_point_y])
            self.look_ahead_point = np.array([look_ahead_x, look_ahead_y])
            self.look_ahead_angle = look_ahead_angle

        lambda_PA = 1
        lambda_CA = 1

        #Collision avoidance reward
        reward_collision_avoidance = 0
        if self.obstacles == []:
            reward_collision_avoidance = 0
            # lambda_PA = 1
            # lambda_CA = 1 - lambda_PA
        else:
            # drone_closest_obs_dist_x = self.invm1to1(obs[8], 0, self.screen_width)
            # drone_closest_obs_dist_y = self.invm1to1(obs[9], 0, self.screen_height)
            drone_closest_obs_dist = self.invm1to1(obs[8], 0, np.sqrt(self.screen_width**2 + self.screen_height**2))
            s_drone_closest_obs_angle = obs[9]
            c_drone_closest_obs_angle = obs[10]

            #These couldve been used in the reward fcn as well, but per now mainly used to keep structure of obs and let agent see more obstacles
            next_closest_obs_dist = self.invm1to1(obs[11], 0, np.sqrt(self.screen_width**2 + self.screen_height**2))
            s_next_closest_obs_angle = obs[12]
            c_next_closest_obs_angle = obs[13]

            n2_closest_obs_dist = self.invm1to1(obs[14], 0, np.sqrt(self.screen_width**2 + self.screen_height**2))
            s_n2_closest_obs_angle = obs[15]
            c_n2_closest_obs_angle = obs[16]

            drone_closest_obs_angle = (np.arctan2(s_drone_closest_obs_angle, c_drone_closest_obs_angle) + 2*np.pi)%(2*np.pi) #range 0 to 2pi

            angle_diff = abs(np.rad2deg((drone_closest_obs_angle - drone_vel_angle + np.pi) % (2 * np.pi) - np.pi))
            if self.render_sim is True:
                self.obs_angle = drone_closest_obs_angle

            danger_range = self.danger_range
            danger_angle = self.danger_angle 
            inv_abs_min_rew = self.abs_inv_CA_min_rew 
            #This scaling determines the inverse of minimum reward for the range and angle reward
            #i.e now the minimum reward is -3/2 = -1.5. Could make it more intuitive by doing inv in equation below, but saves the inversion operation as it is done here.

            #Update so the lambda_path_adherance variable is dynamically lowered the closer the drone is to an obstacle 
            #TODO make hyperparameter to toggle this
            if (drone_closest_obs_dist < danger_range):
                lambda_PA = (drone_closest_obs_dist/danger_range)/2
                if lambda_PA < 0: lambda_PA = 0
                lambda_CA = 1-lambda_PA

            reward_collision_avoidance = 0
            if (drone_closest_obs_dist < danger_range) and (angle_diff < danger_angle):
                range_rew = -(((danger_range+inv_abs_min_rew*danger_range)/(drone_closest_obs_dist+inv_abs_min_rew*danger_range)) -1)
                angle_rew = -(((danger_angle+inv_abs_min_rew*danger_angle)/(angle_diff+inv_abs_min_rew*danger_angle)) -1)
                reward_collision_avoidance = (range_rew + angle_rew)*2
                self.draw_red_velocity = True
                self.draw_orange_obst_vec = True
            elif drone_closest_obs_dist <danger_range:
                range_rew = -(((danger_range+inv_abs_min_rew*danger_range)/(drone_closest_obs_dist+inv_abs_min_rew*danger_range)) -1)
                angle_rew = -(((danger_angle+inv_abs_min_rew*danger_angle)/(angle_diff+inv_abs_min_rew*danger_angle)) -1)
                if angle_rew > 0: angle_rew = 0 #In this case the angle reward may become positive as anglediff may !< danger_angle
                reward_collision_avoidance = range_rew + angle_rew
                self.draw_red_velocity = False
                self.draw_orange_obst_vec = True
            else:
                reward_collision_avoidance = 0
                self.draw_red_velocity = False
                self.draw_orange_obst_vec = False
        # print('reward_collision_avoidance', reward_collision_avoidance)

        #Path adherence reward
        closest_point_on_prepath = np.array([closest_point_x, closest_point_y])
        drone_pos = np.array([drone_pos_x, drone_pos_y])
        dist_from_path = np.linalg.norm(closest_point_on_prepath - drone_pos)
        reward_path_adherence = -(2*(np.clip(dist_from_path, 0, self.PA_band_edge) / self.PA_band_edge) - 1)*self.PA_scale 
        # print('\nreward_path_adherence', reward_path_adherence)

        #Path progression reward
        reward_path_progression = 0
        velocity = np.sqrt(drone_vel_x**2 + drone_vel_y**2)
        scaled_velocity = velocity*self.PP_vel_scale #Velocity can reach ish 1000 so by scaling it down before multiplying with cos a larger range of velocity values is reached
        vel_LA_diff = abs((look_ahead_angle - drone_vel_angle + np.pi) % (2 * np.pi) - np.pi)
        reward_path_progression = np.cos(vel_LA_diff)*scaled_velocity 
        reward_path_progression = np.clip(reward_path_progression, self.PP_rew_min, self.PP_rew_max)
        # print("reward path progression", reward_path_progression)

        #Collision reward
        reward_collision = 0
        end_cond_1 = False
        if self.space.collison:
            reward_collision = self.rew_collision
            end_cond_1 = True

        #Reward for reaching the end
        reach_end_reward = 0
        end_cond_2 = False
        if np.abs(target_dist_x) < self.reach_end_radius and np.abs(target_dist_y) < self.reach_end_radius:
            end_cond_2 = True
            reach_end_reward = self.rew_reach_end
        
        #Reward for alpha angle too aggressive
        agressive_alpha_reward = 0
        end_cond_5 = False
        if drone_alpha > self.AA_band:
            agressive_alpha_reward = -np.sin(drone_alpha)
        if drone_alpha < -self.AA_band:
            agressive_alpha_reward = np.sin(drone_alpha)
        if np.abs(drone_alpha)>=self.AA_angle:
            agressive_alpha_reward = self.rew_AA
            end_cond_5 = True

        #Stops episode, when time is up
        end_cond_4 = False
        if self.current_time_step == self.max_time_steps:
            end_cond_4 = True

        reward = agressive_alpha_reward + reward_path_adherence*lambda_PA + reward_path_progression + reward_collision + reward_collision_avoidance*lambda_CA + reach_end_reward
        # print(reward)

        self.info['reward'] = reward
        self.info['collision_avoidance_reward'] = reward_collision_avoidance
        self.info['path_adherence'] = reward_path_adherence
        self.info["path_progression"] = reward_path_progression
        self.info['collision_reward'] = reward_collision
        self.info['reach_end_reward'] = reach_end_reward
        self.info['agressive_alpha_reward'] = agressive_alpha_reward
        self.info['env_steps'] = self.current_time_step

        if end_cond_1 or end_cond_2 or end_cond_4 or end_cond_5:
            self.done = True
            if end_cond_1: print("Collision")
            if end_cond_2: print("Reached final waypoint")
            if end_cond_4: print("Time is up")
            if end_cond_5: print("Alpha angle too aggressive")

        return obs, reward, self.done, self.info

    def get_obstacle_distances(self,k):

        closest_distance = np.inf
        closest_obs_dict = {}

        for i, obstacle in enumerate(self.obstacles):
            distance,_,_ = self.distance_between_shapes(self.drone.frame_shape, obstacle.shape)
            closest_obs_dict[i] = distance

        #Sort dictionary by value so closest obstacle is first
        sorted_obs_dict = {k: v for k, v in sorted(closest_obs_dict.items(), key=lambda item: item[1])}

        #Get the k closest obstacles by slicing dictionary
        sorted_obs_dict = dict(list(sorted_obs_dict.items())[0:k])

        return sorted_obs_dict

    def get_observation(self):
        # Drone velocities
        velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0))
        # velocity_x = np.clip(velocity_x/1330, -1, 1)
        # velocity_y = np.clip(velocity_y/1330, -1, 1)
        #Assume 1330 is max velocity of drone (it can be larger, but requires a lot of force/time to accelerate to that speed)
        velocity_x = self.m1to1(velocity_x, -1330, 1330)
        velocity_y = self.m1to1(velocity_y, -1330, 1330)

        #Drone angular velocity
        omega = self.drone.frame_shape.body.angular_velocity
        omega = np.clip(omega/11.7, -1, 1)

        #Drone angle
        alpha = self.drone.frame_shape.body.angle
        alpha = (alpha/(np.pi)) #scales from -1 to 1

        x, y = self.drone.frame_shape.body.position
        #May keep it to reward reaching the end or reward path progression
        target_distance_x = self.m1to1(self.x_target-x, 0, self.screen_width)
        target_distance_y = self.m1to1(self.y_target-y, 0, self.screen_height)

        #Position of drone
        pos_x = self.m1to1(x, 0, self.screen_width)
        pos_y = self.m1to1(y, 0, self.screen_height)

        alphadrone = self.drone.frame_shape.body.angle


        if self.obstacles == []:
            closest_obs_distance = 1
            sin_closest_obs_angle = 0
            cos_closest_obs_angle = 0
            next_closest_obs_distance = 1
            next_sin_closest_obs_angle = 0
            next_cos_closest_obs_angle = 0
            n2_closest_obs_distance = 1
            n2_sin_closest_obs_angle = 0
            n2_cos_closest_obs_angle = 0
        else:
            #Distance to closest obstacle old still here to get index of closest obstacle
            closests_obstacles = self.get_obstacle_distances(self.k_obs)
            indexes = list(closests_obstacles.keys())
            euc_dists = list(closests_obstacles.values())

            self.closest_obs_index = indexes[0]

            k_normed_obs_dists = []
            k_sin_obs_angles = []
            k_cos_obs_angles = []

            #Assuming k = 3 always
            closest_obs_distance = 1
            sin_closest_obs_angle = 0
            cos_closest_obs_angle = 0
            next_closest_obs_distance = 1
            next_sin_closest_obs_angle = 0
            next_cos_closest_obs_angle = 0
            n2_closest_obs_distance = 1
            n2_sin_closest_obs_angle = 0
            n2_cos_closest_obs_angle = 0

            for i in range(self.k_obs):
                k_normed_obs_dists.append(self.m1to1(euc_dists[i], 0, np.sqrt(self.screen_width**2 + self.screen_height**2)))
                angle = np.arctan2(y - self.obstacles[indexes[i]].y_pos, x - self.obstacles[indexes[i]].x_pos) #range -pi to pi
                angle = ssa(angle - alphadrone - np.pi)
                k_sin_obs_angles.append(np.sin(angle))
                k_cos_obs_angles.append(np.cos(angle))
            
            if len(self.obstacles) == 1:
                closest_obs_distance = k_normed_obs_dists[0]
                sin_closest_obs_angle = k_sin_obs_angles[0]
                cos_closest_obs_angle = k_cos_obs_angles[0]
            elif len(self.obstacles) == 2:
                closest_obs_distance = k_normed_obs_dists[0]
                sin_closest_obs_angle = k_sin_obs_angles[0]
                cos_closest_obs_angle = k_cos_obs_angles[0]
                next_closest_obs_distance = k_normed_obs_dists[1]
                next_sin_closest_obs_angle = k_sin_obs_angles[1]
                next_cos_closest_obs_angle = k_cos_obs_angles[1]
            else:
                closest_obs_distance = k_normed_obs_dists[0]
                sin_closest_obs_angle = k_sin_obs_angles[0]
                cos_closest_obs_angle = k_cos_obs_angles[0]
                next_closest_obs_distance = k_normed_obs_dists[1]
                next_sin_closest_obs_angle = k_sin_obs_angles[1]
                next_cos_closest_obs_angle = k_cos_obs_angles[1]
                n2_closest_obs_distance = k_normed_obs_dists[2]
                n2_sin_closest_obs_angle = k_sin_obs_angles[2]
                n2_cos_closest_obs_angle = k_cos_obs_angles[2]
 
            #OLD
            # closest_obs_distance = euc_dists[0]
            # closest_obs_x_dist = x_dists[0]
            # closest_obs_y_dist = ydists[0]
            # # closest_obs_distance,closest_obs_x_dist,closest_obs_y_dist = self.distance_between_shapes(self.drone.frame_shape, self.obstacles[self.closest_obs_index].shape)
            # closest_obs_x_dist = self.m1to1(closest_obs_x_dist, 0, self.screen_width)
            # closest_obs_y_dist = self.m1to1(closest_obs_y_dist, 0, self.screen_height)
            # closest_obs_distance = self.m1to1(closest_obs_distance, 0, np.sqrt(self.screen_width**2 + self.screen_height**2))
            # #TODO decide if x and y necessary or if distance + angle is enough

            # #Angle to closest obstacle
            # closest_obs_angle = np.arctan2(y - self.obstacles[self.closest_obs_index].y_pos, x - self.obstacles[self.closest_obs_index].x_pos) #range -pi to pi
            # #body frame
            # closest_obs_angle = closest_obs_angle - alphadrone - np.pi
            # closest_obs_angle = ssa(closest_obs_angle)
            # sin_closest_obs_angle = np.sin(closest_obs_angle) #Pass out the sin and cos rather than just the angle to avoid jumps from -1 to 1.
            # cos_closest_obs_angle = np.cos(closest_obs_angle)

        #Velocity angle
        velocity_x_, velocity_y_ = self.drone.frame_shape.body.velocity_at_local_point((0, 0)) #Velocity in body frame
        velocity_angle_b_hori = np.arctan2(velocity_y_, velocity_x_) #range -pi to pi
        velocity_angle_b = ssa(velocity_angle_b_hori - alphadrone)
        sin_velocity_angle_b = np.sin(velocity_angle_b)
        cos_velocity_angle_b = np.cos(velocity_angle_b)

        #Closest point on path position
        closest_point = self.predef_path.get_closest_position([x,y])
        closest_point_x = closest_point[0]
        closest_point_y = closest_point[1]
        closest_point_x = self.m1to1(closest_point_x, 0, self.screen_width)
        closest_point_y = self.m1to1(closest_point_y, 0, self.screen_height)

        #Lookahead point position
        lookahead_point = self.predef_path.get_lookahead_point([x,y], self.lookahead)
        #Lock LA point to end goal once reached
        if np.abs(lookahead_point[0] - self.wps[-1][0]) < 10 and np.abs(lookahead_point[1] - self.wps[-1][1]) < 10:
            self.LA_in_last_wp = True
        if self.LA_in_last_wp is True:
            lookahead_point = self.wps[-1]
            lookahead_point_x = lookahead_point[0]
            lookahead_point_y = lookahead_point[1]
        else:
            lookahead_point_x = lookahead_point[0]
            lookahead_point_y = lookahead_point[1]
        lookahead_point_x = self.m1to1(lookahead_point_x, 0, self.screen_width)
        lookahead_point_y = self.m1to1(lookahead_point_y, 0, self.screen_height)

        #Angle between drone and lookahead point
        LA_body = np.matmul(R_w_b(alphadrone), lookahead_point - np.array([x,y]))
        look_ahead_angle_b_hori = np.arctan2(LA_body[1],LA_body[0]) #range -pi to pi
        look_ahead_angle = ssa(look_ahead_angle_b_hori - alphadrone)
        s_look_ahead_angle = np.sin(look_ahead_angle)
        c_look_ahead_angle = np.cos(look_ahead_angle)
        
        return np.array([velocity_x, velocity_y, 
                         omega, alpha, 
                         target_distance_x, target_distance_y, 
                         pos_x, pos_y,
                         closest_obs_distance,sin_closest_obs_angle,cos_closest_obs_angle, next_closest_obs_distance, next_sin_closest_obs_angle, next_cos_closest_obs_angle, n2_closest_obs_distance, n2_sin_closest_obs_angle, n2_cos_closest_obs_angle,
                         sin_velocity_angle_b,cos_velocity_angle_b,
                         closest_point_x,closest_point_y,
                         lookahead_point_x,lookahead_point_y,s_look_ahead_angle,c_look_ahead_angle])
    
    def render(self, mode='human', close=False):
        if self.render_sim is False: return

        pygame_events(self.space, self, self.change_target)
        self.screen.fill((243, 243, 243))

        #Debugging --------------------
        #Checking the normal vector of the path
        # for v in self.vec:
        #     pygame.draw.line(self.screen, (170, 0, 170), (v[0], self.screen_height-v[1]), (v[2], self.screen_height-v[3]), 4)
        #Debugging --------------------

        #Draw the reward  as text in top left corner
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Total reward: ' + str(round(self.info['reward'], 2)), True, (0, 0, 0), (243, 243, 243))
        textRect = text.get_rect()
        textRect.topleft = (0, 0)
        self.screen.blit(text, textRect)
        #Draw the collision avoidance reward below the reward
        text = font.render('Collision avoidance: ' + str(round(self.info['collision_avoidance_reward'], 2)), True, (0, 0, 0), (243, 243, 243))
        textRect = text.get_rect()
        textRect.topleft = (0, 32)
        self.screen.blit(text, textRect)
        #Draw the path adherence reward below the collision avoidance reward
        text = font.render('Path adherence: ' + str(round(self.info['path_adherence'], 2)), True, (0, 0, 0), (243, 243, 243))
        textRect = text.get_rect()
        textRect.topleft = (0, 64)
        self.screen.blit(text, textRect)
        #Draw the path progression reward below the path adherence reward
        text = font.render('Path progression: ' + str(round(self.info['path_progression'], 2)), True, (0, 0, 0), (243, 243, 243))
        textRect = text.get_rect()
        textRect.topleft = (0, 96)
        self.screen.blit(text, textRect)
        #Draw the agressive alpha reward below the path progression reward
        text = font.render('Agressive alpha: ' + str(round(self.info['agressive_alpha_reward'], 2)), True, (0, 0, 0), (243, 243, 243))
        textRect = text.get_rect()
        textRect.topleft = (0, 128)
        self.screen.blit(text, textRect)

        #Drawing predefined path
        predef_path_coords = self.predef_path.get_path_coord()
        predef_path_coords = [(x, self.screen_height-y) for x, y in predef_path_coords]
        pygame.draw.aalines(self.screen, (0, 0, 0), False, predef_path_coords)

        #Draw first wp:
        pygame.draw.circle(self.screen, (0, 0, 0), (self.wps[0][0], self.screen_height-self.wps[0][1]), 5)
        #Draw final wp:
        pygame.draw.circle(self.screen, (0, 0, 0), (self.wps[-1][0], self.screen_height-self.wps[-1][1]), 5)

        #Drawing closest point on path
        closest_point = (self.closest_point[0], self.screen_height-self.closest_point[1])
        pygame.draw.circle(self.screen, (0, 0, 255), closest_point, 5)

        drone_x, drone_y = self.drone_pos
        alpha = self.drone_alpha

        #Drawing vector between drone and lookahead point and angle
        pygame.draw.line(self.screen, (0, 150, 150), (drone_x, self.screen_height-drone_y), (self.look_ahead_point[0], self.screen_height-self.look_ahead_point[1]), 4)
        pygame.draw.circle(self.screen, (0, 150, 150), (self.look_ahead_point[0], self.screen_height-self.look_ahead_point[1]), 5)
        pygame.draw.arc(self.screen, (0, 150, 150), (drone_x-50*2, self.screen_height-drone_y-50*2, 100*2, 100*2), alpha, self.look_ahead_angle, 3)

        #Drawing the velocity vector of the drone and velocity angle
        velocity_x, velocity_y = self.drone_vel
        if self.draw_red_velocity is True:
            pygame.draw.line(self.screen, (255, 0, 0), (drone_x, self.screen_height-drone_y), (drone_x+velocity_x, self.screen_height-(drone_y+velocity_y)), 4)
            pygame.draw.arc(self.screen, (255, 0, 0), (drone_x-50, self.screen_height-drone_y-50, 100, 100), alpha, self.vel_angle, 3)
        else:
            pygame.draw.line(self.screen, (0, 0, 0), (drone_x, self.screen_height-drone_y), (drone_x+velocity_x, self.screen_height-(drone_y+velocity_y)), 4)
            pygame.draw.arc(self.screen, (0, 0, 0), (drone_x-50, self.screen_height-drone_y-50, 100, 100), alpha, self.vel_angle, 3)

        #Drawing the vector from drone to nearest obstacle and angle
        if self.obstacles != [] and self.draw_orange_obst_vec is False:
            closest_obs = self.obstacles[self.closest_obs_index]
            pygame.draw.line(self.screen, (0, 255, 0), (drone_x, self.screen_height-drone_y), (closest_obs.x_pos, self.screen_height-closest_obs.y_pos), 4)
            pygame.draw.arc(self.screen, (0, 255, 0), (drone_x-25, self.screen_height-drone_y-25, 50, 50), alpha, self.obs_angle, 3)
        elif self.obstacles != [] and self.draw_orange_obst_vec is True:
            closest_obs = self.obstacles[self.closest_obs_index]
            pygame.draw.line(self.screen, (255, 165, 0), (drone_x, self.screen_height-drone_y), (closest_obs.x_pos, self.screen_height-closest_obs.y_pos), 4)
            pygame.draw.arc(self.screen, (255, 165, 0), (drone_x-25, self.screen_height-drone_y-25, 50, 50), alpha, self.obs_angle, 3)

        #Drawing drone's shade
        if len(self.path_drone_shade):
            for shade in self.path_drone_shade:
                image_rect_rotated = pygame.transform.rotate(self.shade_image, shade[2]*180.0/np.pi)
                shade_image_rect = image_rect_rotated.get_rect(center=(shade[0], self.screen_height-shade[1]))
                self.screen.blit(image_rect_rotated, shade_image_rect)

        #Draws all pymunk objects
        self.space.debug_draw(self.draw_options)

        #Drawing vectors of motor forces
        vector_scale = 0.05
        l_x_1, l_y_1 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, 0))
        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.force_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (l_x_1, self.screen_height-l_y_1), (l_x_2, self.screen_height-l_y_2), 4)

        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.left_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (l_x_1, self.screen_height-l_y_1), (l_x_2, self.screen_height-l_y_2), 4)

        r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 0))
        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.force_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (r_x_1, self.screen_height-r_y_1), (r_x_2, self.screen_height-r_y_2), 4)

        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.right_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (r_x_1, self.screen_height-r_y_1), (r_x_2, self.screen_height-r_y_2), 4)

        pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, self.screen_height-self.y_target), 5)

        #Drawing drone's flight path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        if len(self.drop_path) > 2:
            pygame.draw.aalines(self.screen, (255, 0, 0), False, self.drop_path)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        self.__init__(**self.kwargs)
            # self.render_sim, self.render_path, self.render_shade, self.drone_shade_distance,
            #           self.max_time_steps, self.stabilisation_delay, self.change_target, self.initial_throw)
        return self.get_observation()

    def close(self):
        pygame.quit()

    def initial_movement(self):
        if self.initial_throw is True:
            throw_angle = random.random() * 2*np.pi
            throw_force = random.uniform(0, 1500)
            throw = Vec2d(np.cos(throw_angle)*throw_force, np.sin(throw_angle)*throw_force)

            self.drone.frame_shape.body.apply_force_at_world_point(throw, self.drone.frame_shape.body.position)

            throw_rotation = random.uniform(-3000, 3000)
            self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, throw_rotation), (-self.drone_radius, 0))
            self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, -throw_rotation), (self.drone_radius, 0))

            self.space.step(1.0/60)
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()

        else:
            throw_angle = None
            throw_force = None
            throw_rotation = None

        initial_stabilisation_delay = self.stabilisation_delay
        while self.stabilisation_delay != 0:
            self.space.step(1.0/60)
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True: self.render()
            self.stabilisation_delay -= 1

        self.stabilisation_delay = initial_stabilisation_delay

        return {'throw_angle': throw_angle, 'throw_force': throw_force, 'throw_rotation': throw_rotation}
    
    def distance_between_shapes(self,shape1, shape2):
        '''Returns the minimum distance between two pymunk shapes and the x and y distance between the closest points on the shapes'''
        min_distance = float('inf')
        x_dist = 0
        y_dist = 0
        if type(shape2) is pymunk.shapes.Circle:
            for v1 in shape1.get_vertices():
                v1 = v1 + shape1.body.position
                distance = v1.get_distance(shape2.body.position) - shape2.radius
                if distance < min_distance:
                    min_distance = distance
                    x_dist = abs(v1.x - shape2.body.position.x) #TODO sanity check that abs is correct here.
                    y_dist = abs(v1.y - shape2.body.position.y)
            return min_distance, x_dist, y_dist
        else:
            for v1 in shape1.get_vertices():
                for v2 in shape2.get_vertices():
                    distance = v1.get_distance(v2)
                    if distance < min_distance:
                        min_distance = distance
                        x_dist = abs(v1.x - v2.x)
                        y_dist = abs(v1.y - v2.y)
            return min_distance, x_dist, y_dist

    def m1to1(self,value, min, max):
        '''Normalizes a value from the range [min,max] to the range [-1,1]'''
        return 2.0*(value-min)/(max-min) - 1

    def invm1to1(self, value, min, max):
        '''Inverse normalizes a value from the range [-1,1] to the range [min,max]'''
        return (value+1)*(max-min)/2.0 + min

    def add_postion_to_drop_path(self):
        x, y = self.drone.frame_shape.body.position
        self.drop_path.append((x, self.screen_height-y))

    def add_postion_to_flight_path(self):
        x, y = self.drone.frame_shape.body.position
        self.flight_path.append((x, self.screen_height-y))

    def add_drone_shade(self):
        """
        Adds the current position and angle of the drone to the path_drone_shade list, and updates the shade_x and shade_y
        attributes to the current x and y coordinates of the drone.

        Returns:
            None
        """
        x, y = self.drone.frame_shape.body.position
        self.path_drone_shade.append([x, y, self.drone.frame_shape.body.angle])
        self.shade_x = x
        self.shade_y = y

    def change_target_point(self, x, y):
            """
            Changes the target point for the drone to move towards.

            Args:
                x (float): The x-coordinate of the new target point.
                y (float): The y-coordinate of the new target point.
            """
            self.x_target = x
            self.y_target = y

    def debug_path_angle(self): #to save the vectors created from path angle to draw them in render
        vectors = []
        n = 4
        for _ in range(n):
            u_obs = np.random.uniform(0.20*self.predef_path.length,0.90*self.predef_path.length)
            dist = 20
            path_angle = self.predef_path.get_direction_angle(u_obs)
            x,y = self.predef_path.__call__(u_obs)
            vec = np.array([x,y])+dist*np.array([np.cos(path_angle-np.pi/2),np.sin(path_angle-np.pi/2)])
            # vec = np.array([x,y])+dist*np.array([np.cos(path_angle),np.sin(path_angle)])
            vectors.append([x,y,vec[0],vec[1]])
        return vectors