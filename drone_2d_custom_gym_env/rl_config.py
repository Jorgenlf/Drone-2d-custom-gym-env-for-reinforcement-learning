import numpy as np

# This file contains the configuration for the RL agent and the environment

rl_config = {
    'total_timesteps'       : 9000000,
    'ent_coef'              : 0.01,
}

env_train_config = {
    'render_sim'            : False,
    'render_path'           : False,
    'render_shade'          : False,
    'shade_distance'        : 75,
    'n_steps'               : 1100,
    'n_fall_steps'          : 5,
    'change_target'         : False,
    'initial_throw'         : True,
    'random_path_spawn'     : True,
    'path_segment_length'   : 100,
    'n_wps'                 : 12, # Rule of thumb (screen size / 100)-1 
    'screensize_x'          : 1300,
    'screensize_y'          : 1300,
    'lookahead'             : 220,
    'spawn_corners'         : (1,4), #(DL,DR,UL,UR) 1,4 gives all corners 1,1 gives only bottom left corner 4,4 gives only top right corner 
    'danger_range'          : 150,
    'danger_angle'          : 20,
    'abs_inv_CA_min_rew'    : 1/6, #1/2 -> -2 is min reward per CA fcn range and angle --> rangefcn + anglefcn = -4
    'PA_band_edge'          : 40,
    'PA_scale'              : 2,
    'PP_vel_scale'          : 0.08,
    'PP_rew_max'            : 3.5,
    'PP_rew_min'            : -1,
    'rew_collision'         : -50,
    'reach_end_radius'      : 20,
    'rew_reach_end'         : 30,
    'AA_angle'              : np.pi/2,
    'AA_band'               : np.pi/4, 
    'rew_AA'                : -1,
    'use_Lambda'            : True,
    'mode'                  : 'curriculum', # 'curriculum' or 'test' When test nwps and length are fixed
    'scenario'              : 'stage_5', 
}
#Test scenarios
# 'corridor', 
# 'impossible' 
# 'large' 
# 'parallel', 
# 'perpendicular', 
# 'S_parallel', 
# 'S_corridor', 
# Test in different stages of the curriculum learning
# 'stage_1',
# 'stage_2',
# 'stage_3',
# 'stage_4',
# 'stage_5',

#For vizualization purposes
# env_test_config = env_train_config.copy()
# env_test_config['render_sim']       = True
# env_test_config['render_path']      = True
# env_test_config['render_shade']     = True
# env_test_config['initial_throw']    = False
# env_test_config['n_fall_steps']     = 0

#For speed purposes comment out the one above and use the one below
env_test_config = env_train_config.copy()
env_test_config['render_sim']       = False
env_test_config['render_path']      = True
env_test_config['render_shade']     = False
env_test_config['initial_throw']    = False
env_test_config['n_fall_steps']     = 0