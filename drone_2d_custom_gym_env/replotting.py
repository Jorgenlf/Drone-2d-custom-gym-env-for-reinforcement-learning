import json
import pygame
import pymunk
import numpy as np
import os
        
from test_scenarios import create_test_scenario

def red_blue_grad(float):
    '''takes in a float between 0 and 1 and returns a rgb value between red and blue'''
    r = 0
    g = 0
    b = 0

    if float < 0.5:
        r = 255
        b = 255*float*2 
    else:
        r = 255*(1-float)*2
        b = 255
    return (r,g,b)


#Using results to make a plot of the flight paths
scenario = 'S_parallel'
agent_nr = 20
mode = 'test'
test_nr = 5
screen_width = 1300
screen_height = 1300

file_path = 'Tests/agent_'+agent_nr+'/test_'+str(test_nr)+scenario

# Retrieve the flight paths, rewards and collisions from the test
with open(file_path+'/flight_paths') as json_file:
    flight_paths = json.load(json_file)
rewards = np.load(file_path+'/rewards.npy')
collisions = np.load(file_path+'/collisions.npy')

obstacles = []
space = pymunk.Space()
pymunk.pygame_util.positive_y_is_up = True

if scenario == 'perpendicular':
    wps,predef_path,obstacles=create_test_scenario(space,'perpendicular',screen_width,screen_height)
if scenario == 'parallel':
    wps,predef_path,obstacles=create_test_scenario(space,'parallel',screen_width,screen_height)
if scenario == 'S_parallel':
    wps,predef_path,obstacles=create_test_scenario(space,'S_parallel',screen_width,screen_height)
if scenario == 'corridor':
    wps,predef_path,obstacles=create_test_scenario(space,'corridor',screen_width,screen_height)
if scenario == 'S_corridor':
    wps,predef_path,obstacles=create_test_scenario(space,'S_corridor',screen_width,screen_height)
if scenario == 'large':
    wps,predef_path,obstacles=create_test_scenario(space,'large',screen_width,screen_height)
if scenario == 'impossible':
    wps,predef_path,obstacles=create_test_scenario(space,'impossible',screen_width,screen_height)
if scenario == 'stage_1' or scenario == 'stage_2' or scenario == 'stage_3' or scenario == 'stage_4' or scenario == 'stage_5':
    wps,predef_path,obstacles=None,None,None

pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Drone2d Environment")
screen.fill((243, 243, 243))
if mode == 'test':
    #Draw first wp:
    pygame.draw.circle(screen, (0, 0, 0), (wps[0][0], screen_height-wps[0][1]), 5)
    #Draw final wp:
    pygame.draw.circle(screen, (0, 0, 0), (wps[-1][0], screen_height-wps[-1][1]), 5)

    #Drawing predefined path
    predef_path_coords = predef_path.get_path_coord()
    predef_path_coords = [(x, screen_height-y) for x, y in predef_path_coords]
    pygame.draw.aalines(screen, (0, 0, 0), False, predef_path_coords)

    #Draw obstacles:
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
    space.debug_draw(draw_options)

min_rew = np.min(rewards)
max_rew = np.max(rewards)
normd_rews = (rewards-min_rew)/(max_rew-min_rew)
for i, path in enumerate(flight_paths):
    if len(path) > 2: #Some paths may not be drawn if the drone crashes immediately
        color = red_blue_grad(normd_rews[i])
        if collisions[i] == 1:
            pygame.draw.aalines(screen, (255, 0, 0), False, path, 1)
        else:
            pygame.draw.aalines(screen, color, False, path, 1)
else: pass

#Draw a color bar explaining the color coding of the flight paths blue = high reward, red = low reward
for i in range(100):
    pygame.draw.line(screen, red_blue_grad(i/100), (screen_width-100, screen_height-900-i), (screen_width-50, screen_height-900-i), 1)

font = pygame.font.SysFont('Arial', 30)
text = font.render('High reward', True, (0,0,0))
screen.blit(text, (screen_width-140, screen_height-1030))

font = pygame.font.SysFont('Arial', 30)
text = font.render('Low reward', True, (0,0,0))
screen.blit(text, (screen_width-140, screen_height-910))

pygame.display.flip()
file_path = 'Tests/agent_'+agent_nr+'/test_'+str(test_nr)+'plots'
pygame.image.save(screen, file_path+'/'+scenario+'_'+str(agent_nr)+'.png')