"""Waypoint tracking task terms."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjdrone.tasks.hover import mdp as hover_mdp
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from mjlab.utils.lab_api.math import sample_uniform


def _identity_quat(num_envs: int, device: torch.device | str) -> torch.Tensor:
  quat = torch.zeros(num_envs, 4, device=device)
  quat[:, 0] = 1.0
  return quat


def _sample_yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  zeros = torch.zeros_like(yaw)
  return quat_from_euler_xyz(zeros, zeros, yaw)


class WaypointCommand(CommandTerm):
  cfg: "WaypointCommandCfg"

  def __init__(self, cfg: "WaypointCommandCfg", env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.target_tank: Entity = env.scene[cfg.target_tank_name]
    self.target_body_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
    self.total_reached = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["altitude_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["waypoints_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos_w

  def _update_metrics(self) -> None:
    pos_w = self.robot.data.root_link_pos_w
    error_w = self.target_pos_w - pos_w
    distance = torch.norm(error_w, dim=-1)
    self.metrics["position_error"] = distance
    self.metrics["altitude_error"] = error_w[:, 2].abs()
    self.metrics["waypoints_reached"] = self.total_reached

  def _sample_target_body_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
    radius = sample_uniform(
      self.cfg.target_position_range.radius[0],
      self.cfg.target_position_range.radius[1],
      (len(env_ids),),
      device=self.device,
    )
    angle = sample_uniform(-torch.pi, torch.pi, (len(env_ids),), device=self.device)
    sampled = torch.stack(
      [
        torch.cos(angle) * radius,
        torch.sin(angle) * radius,
        torch.zeros(len(env_ids), device=self.device),
      ],
      dim=-1,
    )
    return sampled + self._env.scene.env_origins[env_ids]

  def _sample_target_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
    headings = torch.tensor(
      [0.0, 0.0, torch.pi, 0.5 * torch.pi, -0.5 * torch.pi],
      device=self.device,
    )
    choices = headings[torch.randint(0, len(headings), (len(env_ids),), device=self.device)]
    return choices + torch.empty(len(env_ids), device=self.device).uniform_(-0.1, 0.1)

  def _sync_target_tank(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    yaw = self._sample_target_yaw(env_ids)
    pose = torch.cat(
      [self.target_body_pos_w[env_ids], _sample_yaw_quat(yaw)],
      dim=-1,
    )
    self.target_tank.write_mocap_pose_to_sim(pose, env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self.target_body_pos_w[env_ids] = self._sample_target_body_positions(env_ids)
    self.target_pos_w[env_ids] = self.target_body_pos_w[env_ids]
    self.target_pos_w[env_ids, 2] = self.cfg.target_contact_height
    self.active[env_ids] = True
    self._sync_target_tank(env_ids)
    hover_mdp.orient_robot_towards_positions(
      self._env,
      env_ids,
      self.target_body_pos_w[env_ids],
      entity_name=self.cfg.entity_name,
    )

  def _update_command(self) -> None:
    self.active[:] = True


@dataclass(kw_only=True)
class WaypointCommandCfg(CommandTermCfg):
  entity_name: str
  target_tank_name: str
  target_contact_height: float = 0.56

  @dataclass
  class TargetPositionRangeCfg:
    radius: tuple[float, float] = (2.8, 7.4)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  def build(self, env) -> WaypointCommand:
    return WaypointCommand(self, env)


def zero_target_error(env, entity_name: str = "robot") -> torch.Tensor:
  del entity_name
  return torch.zeros(env.num_envs, 3, device=env.device)


def zero_target_height(env, entity_name: str = "robot") -> torch.Tensor:
  del entity_name
  return torch.zeros(env.num_envs, 1, device=env.device)


def target_active(env, command_name: str = "waypoint") -> torch.Tensor:
  command: WaypointCommand = env.command_manager.get_term(command_name)
  return command.active.float().unsqueeze(-1)


def waypoint_progress_reward(
  env,
  command_name: str,
  entity_name: str = "robot",
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  target = env.command_manager.get_command(command_name)
  command: WaypointCommand = env.command_manager.get_term(command_name)
  delta = target - robot.data.root_link_pos_w
  distance = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1.0e-4)
  direction = delta / distance
  progress = torch.sum(robot.data.root_link_lin_vel_w * direction, dim=-1)
  return progress * command.active.float()


def active_goal_penalty(env, command_name: str) -> torch.Tensor:
  command: WaypointCommand = env.command_manager.get_term(command_name)
  return command.active.float()


def waypoint_arrival_bonus(
  env,
  sensor_name: str,
  entity_name: str = "robot",
  lift_off_height: float = 0.22,
) -> torch.Tensor:
  return goal_reached(
    env,
    sensor_name=sensor_name,
    entity_name=entity_name,
    lift_off_height=lift_off_height,
  ).float()


def goal_reached(
  env,
  sensor_name: str,
  entity_name: str = "robot",
  lift_off_height: float = 0.22,
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  has_lifted_off = robot.data.root_link_pos_w[:, 2] > lift_off_height
  return has_lifted_off & torch.any(sensor.data.found > 0, dim=-1)


def post_liftoff_nontarget_contact(
  env,
  contact_sensor_name: str,
  target_contact_sensor_name: str,
  entity_name: str = "robot",
  lift_off_height: float = 0.22,
) -> torch.Tensor:
  any_contact = hover_mdp.post_liftoff_contact(
    env,
    sensor_name=contact_sensor_name,
    entity_name=entity_name,
    lift_off_height=lift_off_height,
  )
  target_contact = goal_reached(
    env,
    sensor_name=target_contact_sensor_name,
    entity_name=entity_name,
    lift_off_height=lift_off_height,
  )
  return any_contact & ~target_contact


def nontarget_contact_penalty(
  env,
  contact_sensor_name: str,
  target_contact_sensor_name: str,
  entity_name: str = "robot",
  lift_off_height: float = 0.22,
) -> torch.Tensor:
  return post_liftoff_nontarget_contact(
    env,
    contact_sensor_name=contact_sensor_name,
    target_contact_sensor_name=target_contact_sensor_name,
    entity_name=entity_name,
    lift_off_height=lift_off_height,
  ).float()


camera_rgb = hover_mdp.camera_rgb
target_position_error_b = hover_mdp.target_position_error_b
target_height_error = hover_mdp.target_height_error
position_tracking_reward = hover_mdp.position_tracking_reward
altitude_tracking_reward = hover_mdp.altitude_tracking_reward
upright_reward = hover_mdp.upright_reward
linear_velocity_l2 = hover_mdp.linear_velocity_l2
angular_velocity_l2 = hover_mdp.angular_velocity_l2
action_rate_l2 = hover_mdp.action_rate_l2
action_l2 = hover_mdp.action_l2
out_of_bounds = hover_mdp.out_of_bounds
