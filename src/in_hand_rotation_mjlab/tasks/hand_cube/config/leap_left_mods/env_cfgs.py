"""LEAP Left hand cube rotation: modified environment configurations.

Each function starts from the baseline config.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from ..leap_left.env_cfgs import leap_left_hand_cube_rotate_env_cfg


def leap_left_hand_cube_rotate_softgate_tol_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Baseline config with the "soft" drift gate and a tolerance of 0.5.

  Only two params of ``rewards["rotate_finite_diff"]`` change:
  ``drift_mode`` from "step" to "soft", and ``drift_soft_tolerance`` from
  0.0 to 0.5. Errors up to half of each threshold get no penalty. The tilt
  error is measured as before (Euler roll and pitch).
  See ``object_yaw_finite_diff_clipped``.
  """
  cfg = leap_left_hand_cube_rotate_env_cfg(play=play)
  params = cfg.rewards["rotate_finite_diff"].params
  params["drift_mode"] = "soft"
  params["drift_soft_tolerance"] = 0.5
  return cfg
