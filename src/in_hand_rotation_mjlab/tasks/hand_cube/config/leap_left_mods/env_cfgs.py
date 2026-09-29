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


WIDE_FRICTION_RANGE = (0.4, 1.8)


def leap_left_hand_cube_rotate_widefriction_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Baseline config with a wider hand-cube friction randomization range.

    Only ``events["dr_shared_contact_friction"].params["friction_range"]``
    changes from (0.6, 1.4) to ``WIDE_FRICTION_RANGE``.
    """
    cfg = leap_left_hand_cube_rotate_env_cfg(play=play)
    cfg.events["dr_shared_contact_friction"].params["friction_range"] = (
        WIDE_FRICTION_RANGE
    )
    return cfg
