"""Environment config for the quadcopter hover task."""

from __future__ import annotations

from mjdrone.assets import (
  ROAD_TILE_RADIUS,
  get_billboard_cfg,
  get_car_cfg,
  get_ground_decor_cfg,
  get_quadcopter_cfg,
  get_target_tank_cfg,
  get_tree_cfg,
)
from . import mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import SiteEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensorCfg, CameraSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


TREE_ENTITY_NAMES = tuple(f"tree_{idx}" for idx in range(18))
CAR_ENTITY_NAMES = tuple(f"car_{idx}" for idx in range(12))
BILLBOARD_ENTITY_NAMES = tuple(f"billboard_{idx}" for idx in range(6))
GROUND_ENTITY_NAME = "ground"
TARGET_TANK_ENTITY_NAME = "target_tank"


def make_hover_env_cfg(
  *,
  play: bool = False,
  num_envs: int = 512,
  image_height: int = 96,
  image_width: int = 160,
) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "imu_lin_acc": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_acc"},
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "imu_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "imu_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "target_error_b": ObservationTermCfg(
      func=mdp.target_position_error_b,
      params={"command_name": "hover"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "target_height_error": ObservationTermCfg(
      func=mdp.target_height_error,
      params={"command_name": "hover"},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "heading_error": ObservationTermCfg(
      func=mdp.hover_heading_error,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=dict(actor_terms),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "camera": ObservationGroupCfg(
      terms={
        "front_camera_rgb": ObservationTermCfg(
          func=mdp.camera_rgb,
          params={"sensor_name": "front_camera"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "rotor_thrust": SiteEffortActionCfg(
      entity_name="robot",
      actuator_names=("rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr"),
      scale=2.4,
      offset=3.0,
      preserve_order=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "hover": mdp.HoverCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 7.0),
      debug_vis=False,
    )
  }

  sensors = (
    BuiltinSensorCfg(
      name="imu_lin_acc",
      sensor_type="accelerometer",
      obj=ObjRef(type="site", name="imu_site", entity="robot"),
    ),
    BuiltinSensorCfg(
      name="imu_lin_vel",
      sensor_type="velocimeter",
      obj=ObjRef(type="site", name="imu_site", entity="robot"),
    ),
    BuiltinSensorCfg(
      name="imu_ang_vel",
      sensor_type="gyro",
      obj=ObjRef(type="site", name="imu_site", entity="robot"),
    ),
    CameraSensorCfg(
      name="front_camera",
      parent_body="robot/base",
      pos=(0.135, 0.0, 0.042),
      quat=(0.5, 0.5, -0.5, -0.5),
      fovy=58.0,
      width=image_width,
      height=image_height,
      data_types=("rgb",),
      use_shadows=False,
      use_textures=True,
      enabled_geom_groups=(0,),
    ),
    ContactSensorCfg(
      name="robot_contact",
      primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.25, 0.25),
          "y": (-0.25, 0.25),
          "z": (-0.01, 0.03),
          "roll": (-0.04, 0.04),
          "pitch": (-0.04, 0.04),
          "yaw": (-0.18, 0.18),
        },
        "velocity_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (-0.1, 0.1),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.1, 0.1),
        },
      },
    ),
    "randomize_visual_scene": EventTermCfg(
      func=mdp.randomize_visual_scene,
      mode="reset",
      params={
        "ground_name": GROUND_ENTITY_NAME,
        "tree_names": TREE_ENTITY_NAMES,
        "car_names": CAR_ENTITY_NAMES,
        "billboard_names": BILLBOARD_ENTITY_NAMES,
        "target_tank_name": TARGET_TANK_ENTITY_NAME,
        "road_tile_radius": ROAD_TILE_RADIUS,
      },
    ),
  }

  rewards = {
    "position_tracking": RewardTermCfg(
      func=mdp.position_tracking_reward,
      weight=4.0,
      params={"command_name": "hover", "std": 0.3},
    ),
    "altitude_tracking": RewardTermCfg(
      func=mdp.altitude_tracking_reward,
      weight=2.0,
      params={"command_name": "hover", "std": 0.15},
    ),
    "upright": RewardTermCfg(func=mdp.upright_reward, weight=1.5),
    "post_liftoff_contact": RewardTermCfg(
      func=mdp.contact_penalty,
      weight=-8.0,
      params={"sensor_name": "robot_contact", "lift_off_height": 0.22},
    ),
    "lin_vel_penalty": RewardTermCfg(func=mdp.linear_velocity_l2, weight=-0.1),
    "ang_vel_penalty": RewardTermCfg(func=mdp.angular_velocity_l2, weight=-0.05),
    "yaw_rate_penalty": RewardTermCfg(func=mdp.yaw_rate_l2, weight=-0.1),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.002),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": 1.2},
    ),
    "post_liftoff_contact": TerminationTermCfg(
      func=mdp.post_liftoff_contact,
      params={"sensor_name": "robot_contact", "lift_off_height": 0.22},
    ),
    "out_of_bounds": TerminationTermCfg(
      func=mdp.out_of_bounds,
      params={"max_xy_error": 7.0, "min_height": -0.05, "max_height": 4.0},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=24.0,
      terrain=TerrainImporterCfg(terrain_type="plane"),
      entities={
        "robot": get_quadcopter_cfg(),
        GROUND_ENTITY_NAME: get_ground_decor_cfg(),
        TARGET_TANK_ENTITY_NAME: get_target_tank_cfg(),
        **{name: get_tree_cfg() for name in TREE_ENTITY_NAMES},
        **{name: get_car_cfg() for name in CAR_ENTITY_NAMES},
        **{name: get_billboard_cfg() for name in BILLBOARD_ENTITY_NAMES},
      },
      sensors=sensors,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(
      contact_sensor_maxmatch=256,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=6,
        ls_iterations=12,
      ),
      njmax=2048,
      nconmax=512,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=5.0,
      elevation=-20.0,
      azimuth=70.0,
    ),
    decimation=4,
    episode_length_s=10.0 if not play else 30.0,
  )
