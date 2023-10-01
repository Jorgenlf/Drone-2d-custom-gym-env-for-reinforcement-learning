from typing import List, Optional, Union
from test_obstacles import Obstacle
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



class testEnv(gym.Env):

    def __init__(self,n_steps = 500):
        
        #Parameters
        self.n_steps = n_steps
        self.window_width = 800
        self.window_height = 800
        
        #Init values
        # self.done = False

        self.obstacles = []

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

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        pymunk.pygame_util.positive_y_is_up = True

        self.obstacles.append(Obstacle(200, 600, "square", 200, 200, (188, 72, 72), self.space))
    
    def render(self, mode='human',close=False):
        
        pygame_events(self.space, self)
        self.screen.fill((243, 243, 243))

        pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(0, 0, 800, 800), 8)
        pygame.draw.rect(self.screen, (33, 158, 188), pygame.Rect(50, 50, 700, 700), 4)
        pygame.draw.rect(self.screen, (142, 202, 230), pygame.Rect(200, 200, 400, 400), 4)

        self.space.debug_draw(self.draw_options) #Draws obstacles as theyre part of the space from pymunk


        pygame.display.flip()
        self.clock.tick(60)
    


