"""LEAP Left hand cube rotation: modified environment configurations.

Each function starts from the baseline config and changes one thing.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from ..leap_left.env_cfgs import leap_left_hand_cube_rotate_env_cfg


def leap_left_hand_cube_rotate_softgate_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Baseline config with the "soft" drift gate on the rotation reward.

    Only ``rewards["rotate_finite_diff"].params["drift_mode"]`` changes
    from "step" to "soft". See ``object_yaw_finite_diff_clipped``.
    """
    cfg = leap_left_hand_cube_rotate_env_cfg(play=play)
    cfg.rewards["rotate_finite_diff"].params["drift_mode"] = "soft"
    return cfg
