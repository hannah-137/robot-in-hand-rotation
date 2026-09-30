from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_left_hand_cube_rotate_history3_env_cfg
from .rl_cfg import leap_left_hand_cube_rotate_history3_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Leap-Left-HandCube-Rotate-History3",
  env_cfg=leap_left_hand_cube_rotate_history3_env_cfg(),
  play_env_cfg=leap_left_hand_cube_rotate_history3_env_cfg(play=True),
  rl_cfg=leap_left_hand_cube_rotate_history3_ppo_cfg(),
)
