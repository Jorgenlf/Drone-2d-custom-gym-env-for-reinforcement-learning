import pymunk
import pymunk.pygame_util
import pygame

#File contaning the class for the obstacles in the environment 
class Obstacle():
    
    def __init__(self, x, y, width, height, color, space) -> None:
        #Parameters
        self.color = color
        self.width = width
        self.height = height

        #body
        obstacle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obstacle_body.position = x, y
        
        #shape
        self.obstacle_shape = pymunk.Poly.create_box(obstacle_body, size=(width, height))
        self.obstacle_shape.color = pygame.Color(color)
        self.obstacle_shape.elasticity = 0.1 #Lil' bounce
        self.obstacle_shape.friction = 0.2 #Some friction
        self.obstacle_shape.collision_type = 2

        space.add(obstacle_body, self.obstacle_shape)
    
    def get_position(self):
        return self.obstacle_shape.body.position
    
    def shape(self):
        return self.width, self.height
        
    def get_color(self):
        return self.color

#RAther make a superclass obstacle and then make a subclass for each shape #TODO

#Make function that generate obstacles in a random way relative to drone start pos and predefined path

#Make functions that generate obstacles in a specific way