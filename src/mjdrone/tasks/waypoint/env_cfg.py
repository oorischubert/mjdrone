"""Environment config for waypoint tracking."""

from __future__ import annotations

from mjdrone.tasks.hover.env_cfg import TARGET_TANK_ENTITY_NAME, make_hover_env_cfg
from mjdrone.tasks.waypoint import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

def make_waypoint_env_cfg(
  *,
  play: bool = False,
  num_envs: int = 512,
  image_height: int = 96,
  image_width: int = 160,
):
  cfg = make_hover_env_cfg(
    play=play,
    num_envs=num_envs,
    image_height=image_height,
    image_width=image_width,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    ContactSensorCfg(
      name="target_tank_contact",
      primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="tank", entity=TARGET_TANK_ENTITY_NAME),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  hover_actor_terms = cfg.observations["actor"].terms
  hover_critic_terms = cfg.observations["critic"].terms

  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "imu_lin_acc": hover_actor_terms["imu_lin_acc"],
      "imu_lin_vel": hover_actor_terms["imu_lin_vel"],
      "imu_ang_vel": hover_actor_terms["imu_ang_vel"],
      "projected_gravity": hover_actor_terms["projected_gravity"],
      "target_error_b": ObservationTermCfg(func=mdp.zero_target_error),
      "target_height_error": ObservationTermCfg(func=mdp.zero_target_height),
      "target_active": ObservationTermCfg(func=mdp.target_active),
      "actions": hover_actor_terms["actions"],
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={
      "imu_lin_acc": hover_critic_terms["imu_lin_acc"],
      "imu_lin_vel": hover_critic_terms["imu_lin_vel"],
      "imu_ang_vel": hover_critic_terms["imu_ang_vel"],
      "projected_gravity": hover_critic_terms["projected_gravity"],
      "target_error_b": ObservationTermCfg(
        func=mdp.target_position_error_b,
        params={"command_name": "waypoint"},
      ),
      "target_height_error": ObservationTermCfg(
        func=mdp.target_height_error,
        params={"command_name": "waypoint"},
      ),
      "target_active": ObservationTermCfg(func=mdp.target_active),
      "actions": hover_critic_terms["actions"],
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.commands = {
    "waypoint": mdp.WaypointCommandCfg(
      entity_name="robot",
      target_tank_name=TARGET_TANK_ENTITY_NAME,
      resampling_time_range=(6.0, 9.0),
      debug_vis=False,
    )
  }

  cfg.rewards = {
    "goal_progress": RewardTermCfg(
      func=mdp.waypoint_progress_reward,
      weight=2.4,
      params={"command_name": "waypoint"},
    ),
    "position_tracking": RewardTermCfg(
      func=mdp.position_tracking_reward,
      weight=1.8,
      params={"command_name": "waypoint", "std": 0.85},
    ),
    "altitude_tracking": RewardTermCfg(
      func=mdp.altitude_tracking_reward,
      weight=0.8,
      params={"command_name": "waypoint", "std": 0.4},
    ),
    "waypoint_arrival": RewardTermCfg(
      func=mdp.waypoint_arrival_bonus,
      weight=25.0,
      params={"sensor_name": "target_tank_contact", "lift_off_height": 0.22},
    ),
    "active_goal_time": RewardTermCfg(
      func=mdp.active_goal_penalty,
      weight=-0.05,
      params={"command_name": "waypoint"},
    ),
    "nontarget_contact": RewardTermCfg(
      func=mdp.nontarget_contact_penalty,
      weight=-12.0,
      params={
        "contact_sensor_name": "robot_contact",
        "target_contact_sensor_name": "target_tank_contact",
        "lift_off_height": 0.22,
      },
    ),
    "upright": RewardTermCfg(func=mdp.upright_reward, weight=1.0),
    "lin_vel_penalty": RewardTermCfg(func=mdp.linear_velocity_l2, weight=-0.04),
    "ang_vel_penalty": RewardTermCfg(func=mdp.angular_velocity_l2, weight=-0.03),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.002),
  }

  cfg.terminations["post_liftoff_contact"] = TerminationTermCfg(
    func=mdp.post_liftoff_nontarget_contact,
    params={
      "contact_sensor_name": "robot_contact",
      "target_contact_sensor_name": "target_tank_contact",
      "lift_off_height": 0.22,
    },
  )
  cfg.terminations["out_of_bounds"] = TerminationTermCfg(
    func=mdp.out_of_bounds,
    params={"max_xy_error": 9.0, "min_height": -0.05, "max_height": 4.5},
  )
  cfg.terminations["goal_reached"] = TerminationTermCfg(
    func=mdp.goal_reached,
    params={"sensor_name": "target_tank_contact", "lift_off_height": 0.22},
  )

  cfg.episode_length_s = 24.0 if not play else 36.0
  return cfg
