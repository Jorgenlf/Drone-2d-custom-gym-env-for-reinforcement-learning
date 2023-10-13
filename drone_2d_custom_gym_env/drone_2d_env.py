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
                 n_steps=500, n_fall_steps=10, change_target=False, initial_throw=True):
        #Rendering booleans
        self.render_sim = render_sim
        self.render_path = render_path
        self.render_shade = render_shade

        #Rendering initialization
        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []
            self.drop_path = []
            self.path_drone_shade = []

        #Pymunk initialization
        self.obstacles = []
        self.init_pymunk()

        #Predefined path generation
        self.wps = []
        self.wps = generate_random_waypoints_2d(5,150,'2d',self.obstacles,self.drone_radius)
        self.predef_path = QPMI2D(self.wps)
        self.waypoint_index = 0
        #Related to the skipping of waypoints which is per now unused
        # self.path_prog = []
        # self.passed_waypoints = np.zeros((1, 2), dtype=np.float32)

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
        self.x_target = self.wps[0][0]
        self.y_target = self.wps[0][1]

        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_observation = np.array([-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
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
        random_x = random.uniform(100, 200)
        random_y = random.uniform(100, 200)
        # x1 = self.wps[0][0]
        # y1 = self.wps[0][1]
        angle_rand = random.uniform(-np.pi/4, np.pi/4)
        self.drone = Drone(random_x, random_y, angle_rand, 20, 100, 0.2, 0.4, 0.4, self.space)

        self.drone_radius = self.drone.drone_radius

        #Generating obstacles
        #Randomly generated obstacles
        # self.obstacles = generate_obstacles(4, self.space) #TODO ensure its not buggy
        #TODO maybe add obstacle on path

        #Hardcoded for testing purposes
        obstacle1 = Obstacle(200, 300, 20, 20, (188, 72, 72), self.space)
        self.obstacles.append(obstacle1)
        obstacle2 = Obstacle(600, 500, 20, 20, (188, 72, 72), self.space)
        self.obstacles.append(obstacle2)
        obstacle3 = Obstacle(400, 400, 20, 20, (188, 72, 72), self.space)
        self.obstacles.append(obstacle3)

    # def PID_controller(self):
    #     '''PID controller for stabilizing the drone'''
    #     alpha_ref = 0
    #     alpha = self.drone.frame_shape.body.angle
    #     alpha_dot = self.drone.frame_shape.body.angular_velocity


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

        #Checking if Waypoint is passed uncertain if this is needed
        # if self.predef_path:
        #     self.prog = self.predef_path.get_closest_u(self.drone.frame_shape.body.position, self.waypoint_index)
        #     self.path_prog.append(self.prog)
            
        #     k = self.predef_path.get_u_index(self.prog)
        #     if k > self.waypoint_index:
        #         print("Passed waypoint {:d}".format(k+1), self.predef_path.waypoints[k], "\tquad position:", self.drone.frame_shape.body.position)
        #         self.passed_waypoints = np.vstack((self.passed_waypoints, self.predef_path.waypoints[k]))
        #         # self.waypoint_index = k 

        obs = self.get_observation()
        drone_vel_x = obs[0]
        drone_vel_y = obs[1]
        drone_omega = obs[2]
        drone_alpha = obs[3]
        target_dist_x = obs[4]
        target_dist_y = obs[5]
        drone_pos_x = obs[6]
        drone_pos_y = obs[7]
        drone_closest_obs_dist = obs[8]
        drone_closest_obs_angle = obs[9]
        drone_vel_angle = obs[10]

        #Calulating reward for following the path #Make this part of the observation? Naah its inside the drone and not irl
        closest_point = self.predef_path.get_closest_position(self.drone.frame_shape.body.position, self.waypoint_index)
        dist_from_path = np.linalg.norm(closest_point - self.drone.frame_shape.body.position)
        reward_path_following = np.clip(np.log(dist_from_path), - np.inf, np.log(10)) / (- np.log(10)) 
        # print('\nreward_path_following', reward_path_following)


        #TODO Update so the lambda_path_following variable is dynamically lowered when the drone is close to an obstacle
        lambda_path_following = 1 

        #Collision reward
        reward_collision = 0
        end_cond_1 = False
        if self.space.collison:
            reward_collision = -10
            end_cond_1 = True

        #Collision avoidance reward #TODO Check for a better way to do this
        sensor_range = 100
        reward_collision_avoidance = 0
        for obstacle in self.obstacles:
            distance = np.sqrt((self.drone.frame_shape.body.position[0]  - obstacle.x_pos)**2 + (self.drone.frame_shape.body.position[1] - obstacle.y_pos)**2)
            if distance < sensor_range: #and drone_vel_angle - drone_closest_obs_angle < 0.5:
                reward_collision_avoidance += -100.0/(distance+0.1)
        # print('\nreward_collision_avoidance', reward_collision_avoidance)
        # print('\nVelAngle',np.degrees((np.pi/2)*drone_vel_angle))
        # print('\nObsAngle',np.degrees((np.pi/2)*drone_closest_obs_angle)) #TODO encoroporate these angles to determine if the drone is going towards the obstacle or away from it

        #Move target to next waypoint
        reach_end_reward = 0
        end_cond_2 = False
        if np.abs(target_dist_x) < 0.10 and np.abs(target_dist_y) < 0.10 and self.waypoint_index < len(self.wps)-1:
            self.x_target = self.wps[self.waypoint_index+1][0]
            self.y_target = self.wps[self.waypoint_index+1][1]
            self.waypoint_index += 1
        elif np.abs(target_dist_x) < 0.10 and np.abs(target_dist_y) < 0.10 and self.waypoint_index == len(self.wps)-1:
            end_cond_2 = True
            reach_end_reward = 10
            #Give reward for this? #TODO

        #Stops episode, when drone is out of range or alpha angle too aggressive
        end_cond_3 = False
        OOB_or_too_aggressive_alpha_reward = 0
        if np.abs(drone_alpha)==1 or np.abs(drone_pos_x)==1 or np.abs(drone_pos_y)==1:
            end_cond_3 = True
            OOB_or_too_aggressive_alpha_reward = -5 #TODO What about this reward incorporate it into the reward function?

        #Stops episode, when time is up
        end_cond_4 = False
        if self.current_time_step == self.max_time_steps:
            end_cond_4 = True

        #Reward close too target
        reward_path_following = np.clip(np.log(dist_from_path), - np.inf, np.log(10)) / (- np.log(10)) 

        reward_close_to_target = np.clip(np.log(abs(target_dist_x)+0.1), - np.inf, np.log(10)) / (- np.log(10)) + np.clip(np.log(abs(target_dist_y)+0.1), - np.inf, np.log(10)) / (- np.log(10))
        #TODO make something more understandable than this^
        #OLD(1.0/(np.abs(target_dist_x)+0.1)) + (1.0/(np.abs(target_dist_y)+0.1)) 
        # print('\nreward_close_to_target', reward_close_to_target)

        reward = OOB_or_too_aggressive_alpha_reward + reward_close_to_target + reward_path_following*lambda_path_following + reward_collision + reward_collision_avoidance + reach_end_reward
        # print(reward)

        self.info['reward'] = reward
        self.info['env_steps'] = self.current_time_step

        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            self.done = True
            if end_cond_1: print("Collision")
            if end_cond_2: print("Reached final waypoint")
            if end_cond_3: print("Out of range or too aggressive alpha angle")
            if end_cond_4: print("Time is up")

        return obs, reward, self.done, self.info

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

        #Distance to target
        x, y = self.drone.frame_shape.body.position

        if x < self.x_target:
            distance_x = np.clip((x/self.x_target) - 1, -1, 0)

        else:
            distance_x = np.clip((-x/(self.x_target-800) + self.x_target/(self.x_target-800)) , 0, 1)

        if y < self.y_target:
            distance_y = np.clip((y/self.y_target) - 1, -1, 0)

        else:
            distance_y = np.clip((-y/(self.y_target-800) + self.y_target/(self.y_target-800)) , 0, 1)
        
        #Position of drone
        pos_x = np.clip(x/400.0 - 1, -1, 1)
        pos_y = np.clip(y/400.0 - 1, -1, 1)

        #Distance to closest obstacle
        closest_distance = 1000
        for i, obstacle in enumerate(self.obstacles):
            distance = np.sqrt((x - obstacle.x_pos)**2 + (y - obstacle.y_pos)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_obs_index = i
        closest_distance = np.clip((closest_distance/400.0) - 1, -1, 1)

        #Angle between drone and closest obstacle
        closest_obs_angle = np.arctan2(y - self.obstacles[closest_obs_index].y_pos, x - self.obstacles[closest_obs_index].x_pos)
        closest_obs_angle = np.clip(closest_obs_angle/(np.pi/2), -1, 1)

        #Angle of velocity vector
        velocity_angle = np.arctan2(velocity_y, velocity_x)
        velocity_angle = np.clip(velocity_angle/(np.pi/2), -1, 1)

        return np.array([velocity_x, velocity_y, omega, alpha, distance_x, distance_y, pos_x, pos_y,closest_distance,closest_obs_angle,velocity_angle])

    def render(self, mode='human', close=False):
        if self.render_sim is False: return

        pygame_events(self.space, self, self.change_target)
        self.screen.fill((243, 243, 243))
        pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(0, 0, 800, 800), 8)
        pygame.draw.rect(self.screen, (33, 158, 188), pygame.Rect(50, 50, 700, 700), 4)
        pygame.draw.rect(self.screen, (142, 202, 230), pygame.Rect(200, 200, 400, 400), 4)

        #Drawing predefined path
        predef_path_coords = self.predef_path.get_path_coord()
        predef_path_coords = [(x, 800-y) for x, y in predef_path_coords]
        pygame.draw.aalines(self.screen, (0, 0, 0), False, predef_path_coords)

        #Drawing waypoints
        for wp in self.wps:
            pygame.draw.circle(self.screen, (0, 0, 0), (wp[0], 800-wp[1]), 5)

        #Drawing drone's shade
        if len(self.path_drone_shade):
            for shade in self.path_drone_shade:
                image_rect_rotated = pygame.transform.rotate(self.shade_image, shade[2]*180.0/np.pi)
                shade_image_rect = image_rect_rotated.get_rect(center=(shade[0], 800-shade[1]))
                self.screen.blit(image_rect_rotated, shade_image_rect)

        self.space.debug_draw(self.draw_options) #Draws obstacles as theyre part of the space from pymunk

        #Drawing vectors of motor forces
        vector_scale = 0.05
        l_x_1, l_y_1 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, 0))
        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.froce_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (l_x_1, 800-l_y_1), (l_x_2, 800-l_y_2), 4)

        l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.left_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (l_x_1, 800-l_y_1), (l_x_2, 800-l_y_2), 4)

        r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 0))
        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.froce_scale*vector_scale))
        pygame.draw.line(self.screen, (179,179,179), (r_x_1, 800-r_y_1), (r_x_2, 800-r_y_2), 4)

        r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.right_force*vector_scale))
        pygame.draw.line(self.screen, (255,0,0), (r_x_1, 800-r_y_1), (r_x_2, 800-r_y_2), 4)

        pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, 800-self.y_target), 5)

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

    def add_postion_to_drop_path(self):
        x, y = self.drone.frame_shape.body.position
        self.drop_path.append((x, 800-y))

    def add_postion_to_flight_path(self):
        x, y = self.drone.frame_shape.body.position
        self.flight_path.append((x, 800-y))

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
