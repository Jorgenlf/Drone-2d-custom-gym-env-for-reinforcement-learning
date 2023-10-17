import pymunk
import pymunk.pygame_util
import pygame
import numpy as np

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
        

def generate_obstacles(n, space):
    obstacles = []
    color = (188, 72, 72)
    for _ in range(n):
        random_size = np.random.randint(10, 80)
        random_x = np.random.randint(200, 600)
        random_y = np.random.randint(200, 600)
        if np.random.choice([True, False]):
            obstacle = Square(random_x, random_y, random_size, color, space)
        else:
            random_height = np.random.randint(10, 80)
            obstacle = Rectangle(random_x, random_y, random_size, random_height, color, space)
        obstacles.append(obstacle)

    return obstacles

# Make function that generates obstacles in a random way relative to drone start pos and predefined path
# Make functions that generate obstacles in a specific way
