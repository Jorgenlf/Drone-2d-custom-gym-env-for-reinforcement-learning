import numpy as np

# This file contains the configuration for the RL agent and the environment

rl_config = {
    'total_timesteps'       : 4000000,
}

env_train_config = {
    'render_sim'            : False,
    'render_path'           : False,
    'render_shade'          : False,
    'shade_distance'        : 75,
    'n_steps'               : 1200,
    'n_fall_steps'          : 5,
    'change_target'         : False,
    'initial_throw'         : True,
    'random_path_spawn'     : True,
    'path_segment_length'   : 100,
    'n_wps'                 : 12,
    'screensize_x'          : 1300,
    'screensize_y'          : 1300,
    'lookahead'             : 220,
    'spawn_corners'         : (1,4), #(DL,DR,UL,UR) 1,4 gives all corners 1,1 gives only bottom left corner 4,4 gives only top right corner 
    'danger_range'          : 150,
    'danger_angle'          : 30,
    'abs_inv_CA_min_rew'    : 2/3,
    'PA_band_edge'          : 50,
    'PA_scale'              : 2,
    'PP_vel_scale'          : 0.1,
    'PP_rew_max'            : 4,
    'PP_rew_min'            : -1,
    'rew_collision'         : -50,
    'reach_end_radius'      : 20,
    'rew_reach_end'         : 30,
    'AA_angle'              : np.pi/2,
    'AA_band'               : np.pi/6, 
    'rew_AA'                : -1,
}

env_test_config = env_train_config.copy()
env_test_config['render_sim'] = True
env_test_config['render_path'] = True
env_test_config['render_shade'] = True
env_test_config['initial_throw'] = False
env_test_config['n_fall_steps'] = 0

# env_test_config = {
#     'render_sim'            : True,
#     'render_path'           : True,
#     'render_shade'          : True,
#     'shade_distance'        : 75,
#     'n_steps'               : 1200,
#     'n_fall_steps'          : 5,
#     'change_target'         : False,
#     'initial_throw'         : False,
#     'random_path_spawn'     : True,
#     'path_segment_length'   : 100,
#     'n_wps'                 : 9,
#     'screensize_x'          : 1000,
#     'screensize_y'          : 1000,
#     'lookahead'             : 200,
#     'spawn_corners'         : (1,4), 
#     'danger_range'          : 150,
#     'danger_angle'          : 30,
#     'abs_inv_CA_min_rew'    : 2/3,
#     'PA_band_edge'          : 50,
#     'PA_scale'              : 2,
#     'PP_vel_scale'          : 0.1,
#     'PP_rew_max'            : 4,
#     'PP_rew_min'            : -1,
#     'rew_collision'         : -50,
#     'reach_end_radius'      : 20,
#     'rew_reach_end'         : 30,
#     'AA_angle'              : np.pi/2,
#     'AA_band'               : np.pi/6,
#     'rew_AA'                : -1, 
# }