"""LEAP Left hand cube rotation: modified environment configurations.

Each function starts from the baseline config.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from ..leap_left.env_cfgs import leap_left_hand_cube_rotate_env_cfg


def leap_left_hand_cube_rotate_softgate_v2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Baseline config with the "soft" drift gate and the "up_axis" tilt error.

  Only two params of ``rewards["rotate_finite_diff"]`` change:
  ``drift_mode`` from "step" to "soft", and ``tilt_metric`` from "euler" to
  "up_axis". See ``object_yaw_finite_diff_clipped``.
  """
  cfg = leap_left_hand_cube_rotate_env_cfg(play=play)
  params = cfg.rewards["rotate_finite_diff"].params
  params["drift_mode"] = "soft"
  params["tilt_metric"] = "up_axis"
  return cfg
