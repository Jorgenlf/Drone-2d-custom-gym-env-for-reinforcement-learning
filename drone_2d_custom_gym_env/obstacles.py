import pymunk
import pymunk.pygame_util
import pygame

#File contaning the class for the obstacles in the environment 
#Create the class for the obstacles
class Obstacle():
    
    def __init__(self, x, y, width, height, color, space) -> None:

        self.obstacle_shape = pymunk.Poly.create_box(None, size=(width, height))

        #Parameters
        self.color = color
        self.width = width
        self.height = height

        obstacle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obstacle_body.position = x, y
        
        self.obstacle_shape.body = obstacle_body
        self.obstacle_shape.color = pygame.Color(color)

        self.obstacle_shape.elasticity = 0
        
        space.add(obstacle_body, self.obstacle_shape)
    
    def get_position(self):
        return self.obstacle_shape.body.position
    
    def shape(self):
        return self.width, self.height
        
    def get_color(self):
        return self.color

#RAther make a superclass obstacle and then make a subclass for each shape #TODO

"""
        elif shape == "circle":
            self.obstacle_shape = pymunk.Circle(None, radius)
        else:
            print("Invalid shape")
            return
"""