"""PPO configs for the modified LEAP Left tasks.

The PPO settings are the baseline settings. Only ``experiment_name`` changes
so each modification logs to its own folder.
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..leap_left.rl_cfg import leap_left_hand_cube_rotate_ppo_cfg


def leap_left_hand_cube_rotate_softgate_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = leap_left_hand_cube_rotate_ppo_cfg()
    cfg.experiment_name = "leap_left_hand_cube_rotate_softgate"
    return cfg
