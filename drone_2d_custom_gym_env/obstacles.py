import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
from predef_path import QPMI2D

#File to generate obstacles in the environment
class Obstacle:
    def __init__(self, x, y, color):
        self.color = color
        self.x_pos = x
        self.y_pos = y
        self.obstacle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.obstacle_body.position = x, y

    def get_position(self):
        return self.obstacle_body.position


class Square(Obstacle):
    def __init__(self, x, y, size, color, space):
        super().__init__(x, y, color)
        self.size = size
        self.diagonal = np.sqrt(2 * size ** 2)
        self.shape = pymunk.Poly.create_box(self.obstacle_body, size=(size, size))
        self.shape.color = pygame.Color(color)
        self.shape.elasticity = 0.1  # Little bounce
        self.shape.friction = 0.2  # Some friction
        self.shape.collision_type = 2
        space.add(self.obstacle_body, self.shape)


class Rectangle(Obstacle):
    def __init__(self, x, y, width, height, color, space):
        super().__init__(x, y, color)
        self.width = width
        self.height = height
        self.diagonal = np.sqrt(width ** 2 - height ** 2)
        self.shape = pymunk.Poly.create_box(self.obstacle_body, size=(width, height))
        self.shape.color = pygame.Color(color)
        self.shape.elasticity = 0.1  # Little bounce
        self.shape.friction = 0.2  # Some friction
        self.shape.collision_type = 2
        space.add(self.obstacle_body, self.shape)

class Circle(Obstacle):
    def __init__(self,x,y,radius,color,space):
        super().__init__(x,y,color)
        self.radius = radius
        self.shape = pymunk.Circle(self.obstacle_body, radius)
        self.shape.color = pygame.Color(color)
        self.shape.elasticity = 0.1
        self.shape.friction = 0.2
        self.shape.collision_type = 2
        space.add(self.obstacle_body, self.shape)
        

def generate_obstacles_around_path(n, space, path:QPMI2D, mean, std):
    obstacles = []
    color = (188, 72, 72)
    num_obstacles = 0
    path_lenght = path.length
    while num_obstacles < n:
        #uniform distribution of length along path
        u_obs = np.random.uniform(0.20*path_lenght,0.90*path_lenght)
        #get path angle at u_obs
        path_angle = path.get_direction_angles(u_obs)
        #Draw a normal distributed random number for the distance from the path
        dist = np.random.normal(mean, std)
        #get x,y coordinates of the obstacle if it were placed on the path
        x,y = path.__call__(u_obs)
        obs_on_path_pos = np.array([x,y])
        #offset the obstacle from the path 90 degrees normal on the path
        obs_pos = obs_on_path_pos + dist*np.array([np.cos(path_angle-np.pi/2),np.sin(path_angle-np.pi/2)])

        obs_size = np.random.uniform(10,50) #uniform distribution of size
        if np.linalg.norm(obs_pos - obs_on_path_pos) > obs_size+10: #10 is a safety margin #TODO determine if obstacles are allowed to overlap
            obs = Circle(obs_pos[0],obs_pos[1],obs_size,color,space)    
            obstacles.append(obs)
            num_obstacles += 1
        else:
            continue

    return obstacles

# Make function that generates obstacles in a random way relative to drone start pos and predefined path
# Make functions that generate obstacles in a specific way
