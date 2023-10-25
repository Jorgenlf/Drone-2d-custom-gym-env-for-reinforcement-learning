from Drone import *
from obstacles import *
from event_handler import *
from predef_path import *

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import os

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

    def __init__(self, render_sim=False, render_path=True, render_shade=True, shade_distance=70,
                 n_steps=900, n_fall_steps=10, change_target=False, initial_throw=True,
                 path_segment_length=120,n_wps=6,screensize_x=800,screensize_y=800):
        #Rendering booleans
        self.render_sim = render_sim
        self.render_path = render_path
        self.render_shade = render_shade

        #Rendering initialization
        self.screen_width = screensize_x
        self.screen_height = screensize_y
        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []
            self.drop_path = []
            self.path_drone_shade = []
            self.draw_red = False
            self.obs_angle = 0
            self.vel_angle = 0
            self.closest_point = np.array([0,0])
            self.chi_d = 0

        #Predefined path generation
        self.wps = []
        self.seg_length = path_segment_length
        self.wps = generate_random_waypoints_2d(n_wps,self.seg_length,'2d')
        self.predef_path = QPMI2D(self.wps)
        self.waypoint_index = 0
        self.lookahead = 75

        #Pymunk initialization
        self.obstacles = []
        self.init_pymunk()

        #Parameters
        self.max_time_steps = n_steps
        self.stabilisation_delay = n_fall_steps
        self.drone_shade_distance = shade_distance
        self.froce_scale = 1000
        self.initial_throw = initial_throw
        self.change_target = change_target

        #Initial values
        self.first_step = True
        self.done = False
        self.info = {}
        self.current_time_step = 0
        self.left_force = -1
        self.right_force = -1

        #Generating target position
        # self.x_target = self.wps[0][0]
        # self.y_target = self.wps[0][1]
        #Make last waypoint the target
        self.x_target = self.wps[-1][0]
        self.y_target = self.wps[-1][1]


        #Action and observation space
        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_observation = np.array([-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1,-1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1], dtype=np.float32)
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

        #Generating obstacles
        n_obs = np.random.normal(1, 4)
        if n_obs < 0: n_obs = 0
        self.obstacles = generate_obstacles_around_path(n_obs, self.space, self.predef_path, 0, 100)
        #self.obstacles = []
        #TODO maybe add obstacle on path


    def step(self, action):
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
            if self.render_sim is True and self.render_shade is True: self.add_drone_shade()
            self.info = self.initial_movement()

        self.left_force = (action[0]/2 + 0.5) * self.froce_scale
        self.right_force = (action[1]/2 + 0.5) * self.froce_scale

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

        #Taking in observations and calculation reward
        obs = self.get_observation()
        drone_vel_x = obs[0]
        drone_vel_y = obs[1]
        drone_omega = obs[2]
        drone_alpha = obs[3]
        target_dist_x = self.invm1to1(obs[4],0,self.screen_width)
        target_dist_y = self.invm1to1(obs[5],0,self.screen_height)
        drone_pos_x = self.invm1to1(obs[6],0,self.screen_width)
        drone_pos_y = self.invm1to1(obs[7],0,self.screen_height)

        drone_vel_angle = obs[11]
        drone_vel_angle = (drone_vel_angle)*np.pi
        if drone_vel_angle < 0: drone_vel_angle += 2*np.pi
        if self.render_sim is True:
            self.vel_angle = drone_vel_angle

        #Update so the lambda_path_adherance variable is dynamically lowered when the drone is close to an obstacle
        #TODO more elegantly?
        lambda_PA = 0.5
        lambda_CA = 1- lambda_PA

        #Collision avoidance reward
        #TODO Check if there is a better way to do this
        reward_collision_avoidance = 0
        if self.obstacles == []:
            reward_collision_avoidance = 0
            lambda_PA = 1
            lambda_CA = 1 - lambda_PA
        else:
            drone_closest_obs_dist_x = self.invm1to1(obs[8], 0, self.screen_width)
            drone_closest_obs_dist_y = self.invm1to1(obs[9], 0, self.screen_height)
            drone_closest_obs_dist = np.sqrt((drone_closest_obs_dist_x)**2 + (drone_closest_obs_dist_y)**2)
            drone_closest_obs_dist = drone_closest_obs_dist - self.obstacles[self.closest_obs_index].radius

            #Somewhat inaccurate as the drone may be oriented such that the distance becomes negative...
            #if drone is lef or right of obstacle, subtract drone width/2 from distance
            # if (drone_pos_x < self.obstacles[self.closest_obs_index].x_pos  or drone_pos_x > self.obstacles[self.closest_obs_index].x_pos) and abs(drone_pos_y - self.obstacles[self.closest_obs_index].y_pos) < (self.drone.height/2 + self.obstacles[self.closest_obs_index].radius):
            #     drone_closest_obs_dist = drone_closest_obs_dist - self.drone.width/2

            # #if the drone is above or below then subtract the drone height/2 from the distance
            # if (drone_pos_y < self.obstacles[self.closest_obs_index].y_pos  or drone_pos_y > self.obstacles[self.closest_obs_index].y_pos) and abs(drone_pos_x - self.obstacles[self.closest_obs_index].x_pos) < (self.drone.width/2 + self.obstacles[self.closest_obs_index].radius):
            #     drone_closest_obs_dist = drone_closest_obs_dist - self.drone.height/2

            drone_closest_obs_angle = obs[10]
            drone_closest_obs_angle = (drone_closest_obs_angle)*np.pi
            if drone_closest_obs_angle < 0: drone_closest_obs_angle += 2*np.pi
            if self.render_sim is True:
                self.obs_angle = drone_closest_obs_angle
            angle_diff = abs(drone_vel_angle - drone_closest_obs_angle)*180/np.pi

            danger_range = 100
            danger_angle = 30

            reward_collision_avoidance = 0
            epsilon = 0.001
            if (drone_closest_obs_dist < danger_range) and (angle_diff < danger_angle):

                reward_collision_avoidance = - ((danger_range/(drone_closest_obs_dist+epsilon)+2) + (180.0/(angle_diff+epsilon)+2))
                if reward_collision_avoidance < 0: reward_collision_avoidance = reward_collision_avoidance/10
                if reward_collision_avoidance <-5: reward_collision_avoidance = -5

                self.draw_red = True
                lambda_CA = 0.65
                lambda_PA = 1- lambda_CA
            elif drone_closest_obs_dist <danger_range:

                reward_collision_avoidance = -(((danger_range/(drone_closest_obs_dist+epsilon))+2) + ((180.0/(angle_diff+epsilon))+2))
                if reward_collision_avoidance < 0: reward_collision_avoidance = reward_collision_avoidance/20 #Divided by 20 to make it less negative when drone is close to obstacle but not on collision course
                if reward_collision_avoidance < -5: reward_collision_avoidance = -5

                self.draw_red = False
                lambda_CA = 0.55
                lambda_PA = 1- lambda_CA
            else:
                reward_collision_avoidance = 0
                self.draw_red = False
        # print('reward_collision_avoidance', reward_collision_avoidance)

        #Path adherence reward
        closest_point_x = self.invm1to1(obs[12], 0, self.screen_width)
        closest_point_y = self.invm1to1(obs[13], 0, self.screen_height)
        if self.render_sim is True:
            self.closest_point = np.array([closest_point_x, closest_point_y])

        closest_point_on_prepath = np.array([closest_point_x, closest_point_y])
        drone_pos = np.array([drone_pos_x, drone_pos_y])
        dist_from_path = np.linalg.norm(closest_point_on_prepath - drone_pos)
        # reward_path_adherence = np.clip(np.log(dist_from_path), - np.inf, np.log(50)) / (-np.log(50)) #Wehther 50 or more pixels away from path, reward is -1
        reward_path_adherence = -(2*(np.clip(dist_from_path, 0, 50) / 50) - 1)
        # print('reward_path_adherence', reward_path_adherence)

        #Path progression reward
        reward_path_progression = 0
        #Alternative 1
        # Using desired heading angle
        # u = self.predef_path.get_closest_u(self.drone.frame_shape.body.position, self.waypoint_index)
        # pi_p = self.predef_path.get_direction_angle(u)
        # chi_d = pi_p - np.arctan(closest_point_x/self.lookahead)
        # if self.render_sim is True:
        #     self.chi_d = chi_d
        # print('chi_d', chi_d*180/np.pi)
        #TODO This must be tested further to be verified

        #Alteraive 2
        #Reward velocity vector parallel with lookahead vector

        #Alternative 3
        #Use closeness to the target i.e. the last waypoint.
        #euclidean distance from first wp to last wp
        path_length_euc = np.sqrt((self.wps[0][0] - self.wps[-1][0])**2 + (self.wps[0][1] - self.wps[-1][1])**2)
        dist_to_target = np.sqrt(target_dist_x**2 + target_dist_y**2)
        reward_path_progression = 2*((-dist_to_target/path_length_euc) + 1) #Linear fcn 0 when drone in start 10 when drone at end.
        # print("reward path progression", reward_path_progression)


        #Collision reward
        reward_collision = 0
        end_cond_1 = False
        if self.space.collison:
            reward_collision = -100
            end_cond_1 = True

        #Reward for reaching the end
        reach_end_reward = 0
        end_cond_2 = False
        if np.abs(target_dist_x) < 20 and np.abs(target_dist_y) < 20:
            end_cond_2 = True
            reach_end_reward = 100

        #Reward for drone out of frame or alpha angle too aggressive #TODO determine if this should be kept
        end_cond_3 = False
        OOB_or_too_aggressive_alpha_reward = 0
        if np.abs(drone_alpha)==1 or drone_pos_x>self.screen_width or drone_pos_x <0 or drone_pos_y > self.screen_height or drone_pos_y < 0:
            end_cond_3 = True
            OOB_or_too_aggressive_alpha_reward = -5

        #Stops episode, when time is up
        end_cond_4 = False
        if self.current_time_step == self.max_time_steps:
            end_cond_4 = True

        reward = OOB_or_too_aggressive_alpha_reward + reward_path_adherence*lambda_PA + reward_path_progression + reward_collision + reward_collision_avoidance*lambda_CA + reach_end_reward
        print(reward)

        self.info['reward'] = reward
        self.info['collision_avoidance_reward'] = reward_collision_avoidance
        self.info['path_adherence'] = reward_path_adherence
        self.info["path_progression"] = reward_path_progression
        self.info['collision_reward'] = reward_collision
        self.info['env_steps'] = self.current_time_step

        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            self.done = True
            if end_cond_1: print("Collision")
            if end_cond_2: print("Reached final waypoint")
            if end_cond_3: print("Out of range or too aggressive alpha angle")
            if end_cond_4: print("Time is up")

        return obs, reward, self.done, self.info

    #Per now unused but will later be called in get_observation to get the distance to the k nearest obstacles
    #Or the obstacles that are inside a sensor range
    # def get_obstacle_distances(self,k):
    #     x, y = self.drone.frame_shape.body.position
    #     obstacle_distances = []
    #     for obstacle in self.obstacles:
    #         distance = np.sqrt((x - obstacle.x_pos)**2 + (y - obstacle.y_pos)**2)
    #         obstacle_distances.append(distance)

    #     obstacle_distances.sort()
    #     obstacle_distances = obstacle_distances[:k]
    #     return obstacle_distances

    def get_observation(self):
        # Drone velocities
        velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0))
        velocity_x = np.clip(velocity_x/1330, -1, 1)
        velocity_y = np.clip(velocity_y/1330, -1, 1)

        #Drone angular velocity
        omega = self.drone.frame_shape.body.angular_velocity
        omega = np.clip(omega/11.7, -1, 1)

        #Drone angle
        alpha = self.drone.frame_shape.body.angle
        alpha = np.clip(alpha/(np.pi/2), -1, 1)

        x, y = self.drone.frame_shape.body.position

        #May keep it to reward reaching the end or reward path progression
        target_distance_x = self.m1to1(self.x_target-x, 0, self.screen_width)
        target_distance_y = self.m1to1(self.y_target-y, 0, self.screen_height)

        #Position of drone
        pos_x = self.m1to1(x, 0, self.screen_width)
        pos_y = self.m1to1(y, 0, self.screen_height)

        alphadrone = self.drone.frame_shape.body.angle

        closest_obs_x_dist = 0
        closest_obs_y_dist = 0
        if self.obstacles == []:
            closest_obs_angle = 0 #TODO check if this will make the drone belive it is close to an obstacle when there is none there...
            closest_obs_x_dist = self.screen_width
            closest_obs_y_dist = self.screen_height
        else:
            #Distance to closest obstacle
            closest_distance = 1000000
            for i, obstacle in enumerate(self.obstacles):
                distance = np.sqrt((x - obstacle.x_pos)**2 + (y - obstacle.y_pos)**2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obs_x_dist = x - obstacle.x_pos
                    closest_obs_y_dist = y - obstacle.y_pos
                    self.closest_obs_index = i
            closest_obs_x_dist = self.m1to1(closest_obs_x_dist, 0, self.screen_width)
            closest_obs_y_dist = self.m1to1(closest_obs_y_dist, 0, self.screen_height)

            #Angle to closest obstacle
            closest_obs_angle = np.arctan2(y - self.obstacles[self.closest_obs_index].y_pos, x - self.obstacles[self.closest_obs_index].x_pos) #range -pi to pi
            #body frame
            closest_obs_angle = closest_obs_angle - alphadrone - np.pi
            if closest_obs_angle < -np.pi: closest_obs_angle += 2*np.pi
            if closest_obs_angle > np.pi: closest_obs_angle -= 2*np.pi
            closest_obs_angle = (closest_obs_angle/(np.pi)) #scales from -1 to 1

        #Velocity angle
        velocity_x_, velocity_y_ = self.drone.frame_shape.body.velocity_at_local_point((0, 0))
        velocity_angle_w = np.arctan2(velocity_y_, velocity_x_) #range -pi to pi
        #body frame
        velocity_angle_b = velocity_angle_w - alphadrone
        if velocity_angle_b < -np.pi: velocity_angle_b += 2*np.pi
        if velocity_angle_b > np.pi: velocity_angle_b -= 2*np.pi
        velocity_angle_b = (velocity_angle_b/(np.pi)) #scales from -1 to 1


        closest_point = self.predef_path.get_closest_position(self.drone.frame_shape.body.position) #TODO debug why this thorws index error in predefined path when drone is close to last wp
        closest_point_x = closest_point[0]
        closest_point_y = closest_point[1]
        closest_point_x = self.m1to1(closest_point_x, 0, self.screen_width)
        closest_point_y = self.m1to1(closest_point_y, 0, self.screen_height)

        return np.array([velocity_x, velocity_y, omega, alpha, target_distance_x, target_distance_y, pos_x, pos_y,closest_obs_x_dist,closest_obs_y_dist,closest_obs_angle,velocity_angle_b,closest_point_x,closest_point_y])

    def render(self, mode='human', close=False):
        if self.render_sim is False: return

        pygame_events(self.space, self, self.change_target)
        self.screen.fill((243, 243, 243))

        #Debugging --------------------
        # for v in self.vec:
        #     pygame.draw.line(self.screen, (170, 0, 170), (v[0], self.screen_height-v[1]), (v[2], self.screen_height-v[3]), 4)
        #Debugging --------------------

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

        #Drawing vector between drone and lookahead point
        # lookahead_point = self.predef_path.get_lookahead_point(self.drone.frame_shape.body.position, self.lookahead)
        # pygame.draw.line(self.screen, (0, 150, 150), (self.drone.frame_shape.body.position[0], self.screen_height-self.drone.frame_shape.body.position[1]), (lookahead_point[0], self.screen_height-lookahead_point[1]), 4)
        # pygame.draw.circle(self.screen, (0, 150, 150), (lookahead_point[0], self.screen_height-lookahead_point[1]), 5)

        drone_x, drone_y = self.drone.frame_shape.body.position #TODO maybe change these for the values gathered form the observation

        #TODO update this drawing if chi is used
        #Drawing the chi_d angle at the drones position
        # alpha = self.drone.frame_shape.body.angle
        #want to start drawing the arc from the ybody axis
        # pygame.draw.arc(self.screen, (0, 150, 150), (drone_x-50*1.2, self.screen_height-drone_y-50*1.2, 100*1.2, 100*1.2), 0, self.chi_d, 3)

        #Drawing the velocity vector of the drone and angle
        velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0)) #TODO maybe change these for the values gathered form the observation

        if self.draw_red is True:
            pygame.draw.line(self.screen, (255, 0, 0), (drone_x, self.screen_height-drone_y), (drone_x+velocity_x, self.screen_height-(drone_y+velocity_y)), 4)
            pygame.draw.arc(self.screen, (255, 0, 0), (drone_x-50, self.screen_height-drone_y-50, 100, 100), 0, self.vel_angle, 3)
        else:
            pygame.draw.line(self.screen, (0, 0, 0), (drone_x, self.screen_height-drone_y), (drone_x+velocity_x, self.screen_height-(drone_y+velocity_y)), 4)
            pygame.draw.arc(self.screen, (0, 0, 0), (drone_x-50, self.screen_height-drone_y-50, 100, 100), 0, self.vel_angle, 3)


        #Drawing the vector from drone to nearest obstacle and angle
        if self.obstacles != []:
            closest_obs = self.obstacles[self.closest_obs_index]
            pygame.draw.line(self.screen, (0, 255, 0), (drone_x, self.screen_height-drone_y), (closest_obs.x_pos, self.screen_height-closest_obs.y_pos), 4)
            pygame.draw.arc(self.screen, (0, 255, 0), (drone_x-25, self.screen_height-drone_y-25, 50, 50), 0, self.obs_angle, 3)

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
        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.froce_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (l_x_1, self.screen_height-l_y_1), (l_x_2, self.screen_height-l_y_2), 4)

        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.left_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (l_x_1, self.screen_height-l_y_1), (l_x_2, self.screen_height-l_y_2), 4)

        r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 0))
        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.froce_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (r_x_1, self.screen_height-r_y_1), (r_x_2, self.screen_height-r_y_2), 4)

        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.right_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (r_x_1, self.screen_height-r_y_1), (r_x_2, self.screen_height-r_y_2), 4)

        pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, self.screen_height-self.y_target), 5)

        #Drawing drone's path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        if len(self.drop_path) > 2:
            pygame.draw.aalines(self.screen, (255, 0, 0), False, self.drop_path)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        self.__init__(self.render_sim, self.render_path, self.render_shade, self.drone_shade_distance,
                      self.max_time_steps, self.stabilisation_delay, self.change_target, self.initial_throw)
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