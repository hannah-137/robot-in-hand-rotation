from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  leap_left_hand_cube_rotate_softgate_env_cfg,
  leap_left_hand_cube_rotate_widefriction_env_cfg,
)
from .rl_cfg import (
  leap_left_hand_cube_rotate_softgate_ppo_cfg,
  leap_left_hand_cube_rotate_widefriction_ppo_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Leap-Left-HandCube-Rotate-SoftGate",
  env_cfg=leap_left_hand_cube_rotate_softgate_env_cfg(),
  play_env_cfg=leap_left_hand_cube_rotate_softgate_env_cfg(play=True),
  rl_cfg=leap_left_hand_cube_rotate_softgate_ppo_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Leap-Left-HandCube-Rotate-WideFriction",
  env_cfg=leap_left_hand_cube_rotate_widefriction_env_cfg(),
  play_env_cfg=leap_left_hand_cube_rotate_widefriction_env_cfg(play=True),
  rl_cfg=leap_left_hand_cube_rotate_widefriction_ppo_cfg(),
)
