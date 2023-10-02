from typing import List, Optional, Union
from test_obstacles import Obstacle
from test_drone import test_Drone
from pymunk import Vec2d
import numpy as np
import pygame
import gym
import pymunk
from gym import spaces


from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import sys

def pygame_events(space, myenv):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

def drone_obstacle_collision(arbiter, space, data):
    print("Drone collided with obstacle")
    space.collison = True
    return True

class testEnv(gym.Env):

    def __init__(self,n_steps = 500):
        
        #Parameters
        self.n_steps = n_steps
        self.window_width = 800
        self.window_height = 800
        self.done = False
        self.collison = False
        
        #Init values
        # self.done = False

        self.init_pygame()

        self.init_pymunk()

        #Must define an action space..
        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_observation = np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)


    def init_pygame(self):
        #Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Test Environment")
        self.clock = pygame.time.Clock()

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000)
        self.space.collison = False

        # Register collision handler
        drone_obstacle_handler = self.space.add_collision_handler(1, 2) #Drone is collision type 1, obstacle is collision type 2
        drone_obstacle_handler.begin = drone_obstacle_collision

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        pymunk.pygame_util.positive_y_is_up = True

        self.drone = test_Drone(400, 700, 10, 20, self.space)

        self.obstacle1 = Obstacle(400, 400, 100, 100, (188, 72, 72), self.space)
    
    def step(self,action):
        
        self.space.step(1.0/60)
        obs = ['p','l','a','c','e','h','o','l','d']
        reward = 0
        self.done = False
        self.info = {}

        if self.space.collison is True:
            reward = -1
            self.done = True

        return obs, reward, self.done, self.info
    
    def render(self, mode='human',close=False):
        
        pygame_events(self.space, self)
        self.screen.fill((243, 243, 243))

        pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(0, 0, 800, 800), 8)
        pygame.draw.rect(self.screen, (33, 158, 188), pygame.Rect(50, 50, 700, 700), 4)
        pygame.draw.rect(self.screen, (142, 202, 230), pygame.Rect(200, 200, 400, 400), 4)

        self.space.debug_draw(self.draw_options) #Draws obstacles as theyre part of the space from pymunk

        pygame.display.flip()
        self.clock.tick(60)
    


