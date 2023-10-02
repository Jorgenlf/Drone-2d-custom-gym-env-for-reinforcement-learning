import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame

#Per now just a ball

class test_Drone():
    def __init__(self,x,y,radius,mass,space):
        #parameters
        self.color = (0,255,0)
        self.radius = radius

        #body
        ball_body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius),body_type=pymunk.Body.DYNAMIC)
        ball_body.position = x, y

        #shape
        self.ball_shape = pymunk.Circle(ball_body, radius)
        self.ball_shape.color = pygame.Color(self.color)
        self.ball_shape.elasticity = 0.9
        self.ball_shape.friction = 0.05
        self.ball_shape.sensor = False #Wether the physics is applied to the object or not if sensor true no physics but makes a collison callback
        self.ball_shape.collision_type = 1

        space.add(ball_body, self.ball_shape)
