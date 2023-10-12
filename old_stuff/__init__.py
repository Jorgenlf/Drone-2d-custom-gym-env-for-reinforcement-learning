from drone_2d_custom_gym_env.drone_2d_env import *
from gym.envs.registration import register
## If the whole folder imported as a package to other files outside the folder the __init__.py file will work as a entry point.
# As I now have messed with the folder/file structure This file does not work as an entry point anymore. And the register command is ran from the eval.py file. and the __init__.py file is not used at all.

register(
    id='drone-2d-custom-v0',
    entry_point='drone_2d_custom_gym_env:Drone2dEnv',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': False,
            'initial_throw': True}
)
