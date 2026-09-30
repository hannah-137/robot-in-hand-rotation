"""LEAP Left hand cube rotation: modified environment configurations.

Each function starts from the baseline config.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from ..leap_left.env_cfgs import leap_left_hand_cube_rotate_env_cfg


def leap_left_hand_cube_rotate_history3_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Baseline config with a shorter actor observation history.

  Only ``observations["actor"].history_length`` changes, from 10 to 3.
  The critic observations do not change.
  """
  cfg = leap_left_hand_cube_rotate_env_cfg(play=play)
  cfg.observations["actor"].history_length = 3
  return cfg
