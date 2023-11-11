
# This file contains the configuration for the RL agent and the environment
rl_config = {
    'total_timesteps'       : 2000000,
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
    'n_wps'                 : 9,
    'screensize_x'          : 1000,
    'screensize_y'          : 1000,
}

env_test_config = {
    'render_sim'            : True,
    'render_path'           : True,
    'render_shade'          : True,
    'shade_distance'        : 75,
    'n_steps'               : 1200,
    'n_fall_steps'          : 5,
    'change_target'         : False,
    'initial_throw'         : False,
    'random_path_spawn'     : True,
    'path_segment_length'   : 100,
    'n_wps'                 : 9,
    'screensize_x'          : 1000,
    'screensize_y'          : 1000,
}