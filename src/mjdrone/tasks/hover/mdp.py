"""Hover task commands, observations, rewards, and terminations."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjdrone.assets import ROAD_TILE_RADIUS, ROAD_TILE_SIZE
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.lab_api.math import quat_apply, quat_inv, sample_uniform
from mjlab.utils.lab_api.math import quat_from_euler_xyz


class HoverCommand(CommandTerm):
  cfg: "HoverCommandCfg"

  def __init__(self, cfg: "HoverCommandCfg", env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["altitude_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos_w

  def _update_metrics(self) -> None:
    pos_w = self.robot.data.root_link_pos_w
    error_w = self.target_pos_w - pos_w
    self.metrics["position_error"] = torch.norm(error_w, dim=-1)
    self.metrics["altitude_error"] = error_w[:, 2].abs()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    lower = torch.tensor(
      [
        self.cfg.target_position_range.x[0],
        self.cfg.target_position_range.y[0],
        self.cfg.target_position_range.z[0],
      ],
      device=self.device,
    )
    upper = torch.tensor(
      [
        self.cfg.target_position_range.x[1],
        self.cfg.target_position_range.y[1],
        self.cfg.target_position_range.z[1],
      ],
      device=self.device,
    )
    sampled = sample_uniform(lower, upper, (len(env_ids), 3), device=self.device)
    self.target_pos_w[env_ids] = sampled + self._env.scene.env_origins[env_ids]

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class HoverCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class TargetPositionRangeCfg:
    x: tuple[float, float] = (-0.45, 0.45)
    y: tuple[float, float] = (-0.45, 0.45)
    z: tuple[float, float] = (0.85, 1.2)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  def build(self, env) -> HoverCommand:
    return HoverCommand(self, env)


def camera_rgb(env, sensor_name: str) -> torch.Tensor:
  """Return RGB images as (B, C, H, W) float tensors in [0, 1]."""
  sensor = env.scene[sensor_name]
  rgb = sensor.data.rgb
  assert rgb is not None, f"Camera '{sensor_name}' has no RGB data."
  return rgb.permute(0, 3, 1, 2).float() / 255.0


def _resolve_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  if env_ids is None or isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  return env_ids.to(device=env.device, dtype=torch.long)


def _sample_yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  zeros = torch.zeros_like(yaw)
  return quat_from_euler_xyz(zeros, zeros, yaw)


def _set_entity_color(
  env,
  env_ids: torch.Tensor,
  entity_name: str,
  geom_name_pattern: str,
  rgba: torch.Tensor,
) -> None:
  entity: Entity = env.scene[entity_name]
  local_ids, _ = entity.find_geoms(geom_name_pattern, preserve_order=True)
  geom_ids = entity.indexing.geom_ids[local_ids].to(env.device, dtype=torch.long)
  env.sim.model.geom_rgba[env_ids[:, None], geom_ids[None, :]] = rgba[:, None, :]


def _rand_range(
  env,
  num_envs: int,
  lower: float,
  upper: float,
) -> torch.Tensor:
  return torch.rand(num_envs, device=env.device) * (upper - lower) + lower


def _rgba_with_alpha(base_rgba: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
  rgba = base_rgba.clone()
  rgba[:, 3] = alpha
  return rgba


def _random_signs(env, num_envs: int) -> torch.Tensor:
  return torch.where(
    torch.rand(num_envs, device=env.device) > 0.5,
    torch.ones(num_envs, device=env.device),
    -torch.ones(num_envs, device=env.device),
  )


def _road_axis_values(env, road_tile_radius: int) -> torch.Tensor:
  return (
    torch.arange(-road_tile_radius, road_tile_radius + 1, device=env.device, dtype=torch.float32)
    * ROAD_TILE_SIZE
  )


def _enforce_min_planar_radius(pos: torch.Tensor, min_radius: float) -> torch.Tensor:
  xy = pos[:, :2]
  radius = torch.norm(xy, dim=-1, keepdim=True)
  safe_radius = torch.clamp(radius, min=1.0e-6)
  fallback = torch.zeros_like(xy)
  fallback[:, 0] = min_radius
  pushed_xy = xy * (min_radius / safe_radius)
  xy = torch.where(radius < min_radius, pushed_xy, xy)
  xy = torch.where(radius < 1.0e-6, fallback, xy)
  return torch.cat([xy, pos[:, 2:3]], dim=-1)


def _avoid_forward_corridor(
  pos: torch.Tensor,
  *,
  x_min: float,
  x_max: float,
  half_width: float,
) -> torch.Tensor:
  x = pos[:, 0]
  y = pos[:, 1]
  inside = (x > x_min) & (x < x_max) & (y.abs() < half_width)
  push_sign = torch.where(y >= 0.0, torch.ones_like(y), -torch.ones_like(y))
  push_sign = torch.where(y.abs() < 1.0e-4, torch.ones_like(y), push_sign)
  adjusted_y = push_sign * (half_width + 0.2)
  y = torch.where(inside, adjusted_y, y)
  return torch.stack([x, y, pos[:, 2]], dim=-1)


def orient_robot_towards_positions(
  env,
  env_ids: torch.Tensor,
  target_pos_w: torch.Tensor,
  *,
  entity_name: str = "robot",
  yaw_jitter: float = 0.12,
) -> None:
  robot: Entity = env.scene[entity_name]
  # During reset events, root state writes update qpos immediately but derived
  # world-frame pose buffers are stale until the next forward() call.
  root_pose_q = robot.data.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
  robot_pos_w = root_pose_q[:, 0:3]
  robot_quat_w = root_pose_q[:, 3:7]
  roll, pitch, _ = euler_xyz_from_quat(robot_quat_w)
  delta = target_pos_w - robot_pos_w
  yaw = torch.atan2(delta[:, 1], delta[:, 0])
  if yaw_jitter > 0.0:
    yaw = yaw + sample_uniform(-yaw_jitter, yaw_jitter, yaw.shape, device=env.device)
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  robot.write_root_link_pose_to_sim(torch.cat([robot_pos_w, quat], dim=-1), env_ids=env_ids)


def _randomize_road_network(
  env,
  env_ids: torch.Tensor,
  ground_name: str,
  road_tile_radius: int,
  road_rgba: torch.Tensor,
  marking_rgba: torch.Tensor,
) -> None:
  num_envs = len(env_ids)
  grid_size = road_tile_radius * 2 + 1
  center = road_tile_radius
  horizontal = torch.rand((num_envs, grid_size, grid_size - 1), device=env.device) < 0.52
  vertical = torch.rand((num_envs, grid_size - 1, grid_size), device=env.device) < 0.52

  # Keep a continuous cross through the center and a denser inner ring.
  horizontal[:, center, :] = True
  vertical[:, :, center] = True
  if center > 0:
    horizontal[:, center - 1, center - 1 : center + 1] = True
    horizontal[:, center + 1, center - 1 : center + 1] = True
    vertical[:, center - 1 : center + 1, center - 1] = True
    vertical[:, center - 1 : center + 1, center + 1] = True

  for row in range(grid_size):
    for col in range(grid_size):
      north_active = vertical[:, row, col] if row < grid_size - 1 else torch.zeros(num_envs, device=env.device, dtype=torch.bool)
      south_active = vertical[:, row - 1, col] if row > 0 else torch.zeros(num_envs, device=env.device, dtype=torch.bool)
      east_active = horizontal[:, row, col] if col < grid_size - 1 else torch.zeros(num_envs, device=env.device, dtype=torch.bool)
      west_active = horizontal[:, row, col - 1] if col > 0 else torch.zeros(num_envs, device=env.device, dtype=torch.bool)
      center_active = north_active | south_active | east_active | west_active

      suffix = f"{row}_{col}"
      _set_entity_color(env, env_ids, ground_name, f"road_center_{suffix}", _rgba_with_alpha(road_rgba, center_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"road_north_{suffix}", _rgba_with_alpha(road_rgba, north_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"road_south_{suffix}", _rgba_with_alpha(road_rgba, south_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"road_east_{suffix}", _rgba_with_alpha(road_rgba, east_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"road_west_{suffix}", _rgba_with_alpha(road_rgba, west_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"lane_north_{suffix}", _rgba_with_alpha(marking_rgba, north_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"lane_south_{suffix}", _rgba_with_alpha(marking_rgba, south_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"lane_east_{suffix}", _rgba_with_alpha(marking_rgba, east_active.float()))
      _set_entity_color(env, env_ids, ground_name, f"lane_west_{suffix}", _rgba_with_alpha(marking_rgba, west_active.float()))


@requires_model_fields("geom_rgba")
def randomize_visual_scene(
  env,
  env_ids: torch.Tensor | slice | None,
  ground_name: str,
  tree_names: tuple[str, ...],
  car_names: tuple[str, ...],
  billboard_names: tuple[str, ...],
  target_tank_name: str | None = None,
  road_tile_radius: int = ROAD_TILE_RADIUS,
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return

  num_envs = len(env_ids)
  origins = env.scene.env_origins[env_ids]

  # Ground colors: keep the scene outdoors instead of the default gray plane.
  grass = torch.stack(
    [
      _rand_range(env, num_envs, 0.16, 0.24),
      _rand_range(env, num_envs, 0.3, 0.42),
      _rand_range(env, num_envs, 0.12, 0.2),
      torch.ones(num_envs, device=env.device),
    ],
    dim=-1,
  )
  grass_inner = torch.stack(
    [
      _rand_range(env, num_envs, 0.18, 0.26),
      _rand_range(env, num_envs, 0.34, 0.46),
      _rand_range(env, num_envs, 0.14, 0.22),
      torch.ones(num_envs, device=env.device),
    ],
    dim=-1,
  )
  road_gray = _rand_range(env, num_envs, 0.14, 0.22)
  road = torch.stack(
    [road_gray, road_gray + 0.01, road_gray + 0.02, torch.ones(num_envs, device=env.device)],
    dim=-1,
  )
  marking = torch.stack(
    [
      _rand_range(env, num_envs, 0.9, 0.98),
      _rand_range(env, num_envs, 0.88, 0.96),
      _rand_range(env, num_envs, 0.78, 0.9),
      torch.ones(num_envs, device=env.device),
    ],
    dim=-1,
  )
  _set_entity_color(env, env_ids, ground_name, "grass_main", grass)
  _set_entity_color(env, env_ids, ground_name, "grass_inner", grass_inner)
  _randomize_road_network(
    env,
    env_ids,
    ground_name,
    road_tile_radius,
    road,
    marking,
  )

  for idx, entity_name in enumerate(tree_names):
    entity: Entity = env.scene[entity_name]
    angle = _rand_range(env, num_envs, -torch.pi, torch.pi)
    radius = _rand_range(env, num_envs, 4.8 + 0.12 * idx, 9.4 + 0.18 * idx)
    pos = torch.stack(
      [
        torch.cos(angle) * radius,
        torch.sin(angle) * radius,
        torch.zeros(num_envs, device=env.device),
      ],
      dim=-1,
    )
    pos = _avoid_forward_corridor(pos, x_min=1.2, x_max=8.4, half_width=1.8)
    yaw = _rand_range(env, num_envs, -torch.pi, torch.pi)
    pose = torch.cat([origins + pos, _sample_yaw_quat(yaw)], dim=-1)
    entity.write_mocap_pose_to_sim(pose, env_ids)

    canopy = torch.stack(
      [
        _rand_range(env, num_envs, 0.16, 0.32),
        _rand_range(env, num_envs, 0.42, 0.68),
        _rand_range(env, num_envs, 0.14, 0.3),
        torch.ones(num_envs, device=env.device),
      ],
      dim=-1,
    )
    trunk = torch.stack(
      [
        _rand_range(env, num_envs, 0.34, 0.46),
        _rand_range(env, num_envs, 0.22, 0.31),
        _rand_range(env, num_envs, 0.12, 0.18),
        torch.ones(num_envs, device=env.device),
      ],
      dim=-1,
    )
    _set_entity_color(env, env_ids, entity_name, "trunk", trunk)
    _set_entity_color(env, env_ids, entity_name, "canopy_.*", canopy)

  car_palette = torch.tensor(
    [
      [0.22, 0.28, 0.42, 1.0],
      [0.22, 0.34, 0.56, 1.0],
      [0.28, 0.34, 0.3, 1.0],
      [0.38, 0.22, 0.22, 1.0],
      [0.44, 0.44, 0.48, 1.0],
      [0.26, 0.22, 0.34, 1.0],
    ],
    device=env.device,
  )
  for idx, entity_name in enumerate(car_names):
    entity = env.scene[entity_name]
    axis_values = _road_axis_values(env, road_tile_radius)
    axis_choice = axis_values[torch.randint(0, len(axis_values), (num_envs,), device=env.device)]
    lane_offset = _random_signs(env, num_envs) * _rand_range(env, num_envs, 0.28, 0.42)
    along = _rand_range(
      env,
      num_envs,
      -(road_tile_radius + 0.35) * ROAD_TILE_SIZE,
      (road_tile_radius + 0.35) * ROAD_TILE_SIZE,
    )
    horizontal_layout = torch.rand(num_envs, device=env.device) > 0.5
    x = torch.where(horizontal_layout, along, axis_choice + lane_offset)
    y = torch.where(horizontal_layout, axis_choice + lane_offset, along)
    pos = torch.stack([x, y, torch.zeros(num_envs, device=env.device)], dim=-1)
    pos = _enforce_min_planar_radius(pos, min_radius=2.25)
    pos = _avoid_forward_corridor(pos, x_min=1.2, x_max=8.2, half_width=1.5)
    forward_choice = torch.rand(num_envs, device=env.device) > 0.5
    horizontal_heading = torch.where(
      forward_choice,
      torch.zeros(num_envs, device=env.device),
      torch.full((num_envs,), torch.pi, device=env.device),
    )
    vertical_heading = torch.where(
      forward_choice,
      torch.full((num_envs,), 0.5 * torch.pi, device=env.device),
      torch.full((num_envs,), -0.5 * torch.pi, device=env.device),
    )
    heading_center = torch.where(horizontal_layout, horizontal_heading, vertical_heading)
    yaw = heading_center + _rand_range(env, num_envs, -0.22, 0.22)
    pose = torch.cat([origins + pos, _sample_yaw_quat(yaw)], dim=-1)
    entity.write_mocap_pose_to_sim(pose, env_ids)

    palette_idx = torch.randint(0, len(car_palette), (num_envs,), device=env.device)
    car_color = car_palette[palette_idx]
    _set_entity_color(env, env_ids, entity_name, "body|roof", car_color)

  panel_palette = torch.tensor(
    [
      [0.9, 0.78, 0.2, 1.0],
      [0.22, 0.56, 0.88, 1.0],
      [0.9, 0.42, 0.18, 1.0],
      [0.24, 0.68, 0.4, 1.0],
    ],
    device=env.device,
  )
  stripe_palette = torch.tensor(
    [
      [0.16, 0.22, 0.72, 1.0],
      [0.78, 0.18, 0.22, 1.0],
      [0.18, 0.18, 0.2, 1.0],
      [0.95, 0.92, 0.86, 1.0],
    ],
    device=env.device,
  )
  for idx, entity_name in enumerate(billboard_names):
    entity = env.scene[entity_name]
    angle = _rand_range(env, num_envs, -torch.pi, torch.pi)
    radius = _rand_range(env, num_envs, 8.4 + 0.2 * idx, 10.6 + 0.3 * idx)
    pos = torch.stack(
      [
        torch.cos(angle) * radius,
        torch.sin(angle) * radius,
        torch.zeros(num_envs, device=env.device),
      ],
      dim=-1,
    )
    facing_yaw = torch.atan2(-pos[:, 1], -pos[:, 0])
    yaw = facing_yaw + _rand_range(env, num_envs, -0.24, 0.24)
    pose = torch.cat([origins + pos, _sample_yaw_quat(yaw)], dim=-1)
    entity.write_mocap_pose_to_sim(pose, env_ids)

    panel_idx = torch.randint(0, len(panel_palette), (num_envs,), device=env.device)
    stripe_idx = torch.randint(0, len(stripe_palette), (num_envs,), device=env.device)
    _set_entity_color(env, env_ids, entity_name, "panel", panel_palette[panel_idx])
    _set_entity_color(env, env_ids, entity_name, "stripe", stripe_palette[stripe_idx])

  if target_tank_name is not None:
    entity = env.scene[target_tank_name]
    radius = _rand_range(env, num_envs, 2.8, 7.4)
    angle = _rand_range(env, num_envs, -torch.pi, torch.pi)
    x = torch.cos(angle) * radius
    y = torch.sin(angle) * radius
    pos = torch.stack(
      [x, y, torch.zeros(num_envs, device=env.device)],
      dim=-1,
    )
    yaw_choices = torch.tensor(
      [0.0, 0.0, torch.pi, 0.5 * torch.pi, -0.5 * torch.pi],
      device=env.device,
    )
    yaw = yaw_choices[torch.randint(0, len(yaw_choices), (num_envs,), device=env.device)]
    yaw = yaw + _rand_range(env, num_envs, -0.1, 0.1)
    pose = torch.cat([origins + pos, _sample_yaw_quat(yaw)], dim=-1)
    entity.write_mocap_pose_to_sim(pose, env_ids)

    _set_entity_color(
      env,
      env_ids,
      target_tank_name,
      "hull|glacis",
      torch.tensor([0.9, 0.74, 0.28, 1.0], device=env.device).repeat(num_envs, 1),
    )
    _set_entity_color(
      env,
      env_ids,
      target_tank_name,
      "turret|turret_top|side_skirt_.*",
      torch.tensor([0.98, 0.82, 0.26, 1.0], device=env.device).repeat(num_envs, 1),
    )
    _set_entity_color(
      env,
      env_ids,
      target_tank_name,
      "barrel_tip",
      torch.tensor([0.08, 0.08, 0.08, 1.0], device=env.device).repeat(num_envs, 1),
    )
    orient_robot_towards_positions(env, env_ids, origins + pos)


def target_position_error_b(env, command_name: str, entity_name: str = "robot") -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  target = env.command_manager.get_command(command_name)
  error_w = target - robot.data.root_link_pos_w
  return quat_apply(quat_inv(robot.data.root_link_quat_w), error_w)


def target_height_error(env, command_name: str, entity_name: str = "robot") -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  target = env.command_manager.get_command(command_name)
  return (target[:, 2:3] - robot.data.root_link_pos_w[:, 2:3]).clamp(-2.0, 2.0)


def hover_heading_error(
  env,
  entity_name: str = "robot",
  target_heading: float = 0.0,
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  error = robot.data.heading_w - target_heading
  wrapped = torch.atan2(torch.sin(error), torch.cos(error))
  return wrapped.unsqueeze(-1)


def position_tracking_reward(
  env,
  command_name: str,
  entity_name: str = "robot",
  std: float = 0.35,
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  target = env.command_manager.get_command(command_name)
  error = torch.sum((target[:, :2] - robot.data.root_link_pos_w[:, :2]) ** 2, dim=-1)
  return torch.exp(-error / (std**2))


def altitude_tracking_reward(
  env,
  command_name: str,
  entity_name: str = "robot",
  std: float = 0.15,
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  target = env.command_manager.get_command(command_name)
  error = (target[:, 2] - robot.data.root_link_pos_w[:, 2]) ** 2
  return torch.exp(-error / (std**2))


def upright_reward(env, entity_name: str = "robot", std: float = 0.3) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  tilt = torch.sum(robot.data.projected_gravity_b[:, :2] ** 2, dim=-1)
  return torch.exp(-tilt / (std**2))


def heading_tracking_reward(
  env,
  entity_name: str = "robot",
  target_heading: float = 0.0,
  std: float = 0.35,
) -> torch.Tensor:
  error = hover_heading_error(env, entity_name=entity_name, target_heading=target_heading).squeeze(-1)
  return torch.exp(-(error**2) / (std**2))


def linear_velocity_l2(env, entity_name: str = "robot") -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  return torch.sum(robot.data.root_link_lin_vel_b**2, dim=-1)


def angular_velocity_l2(env, entity_name: str = "robot") -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  return torch.sum(robot.data.root_link_ang_vel_b**2, dim=-1)


def yaw_rate_l2(env, entity_name: str = "robot") -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  return robot.data.root_link_ang_vel_b[:, 2] ** 2


def post_liftoff_contact(
  env,
  sensor_name: str,
  entity_name: str = "robot",
  lift_off_height: float = 0.22,
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  has_lifted_off = robot.data.root_link_pos_w[:, 2] > lift_off_height
  any_contact = torch.any(sensor.data.found > 0, dim=-1)
  return has_lifted_off & any_contact


def contact_penalty(
  env,
  sensor_name: str,
  entity_name: str = "robot",
  lift_off_height: float = 0.22,
) -> torch.Tensor:
  return post_liftoff_contact(
    env,
    sensor_name=sensor_name,
    entity_name=entity_name,
    lift_off_height=lift_off_height,
  ).float()


def action_rate_l2(env) -> torch.Tensor:
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(delta**2, dim=-1)


def action_l2(env) -> torch.Tensor:
  return torch.sum(env.action_manager.action**2, dim=-1)


def out_of_bounds(
  env,
  entity_name: str = "robot",
  max_xy_error: float = 2.0,
  min_height: float = 0.2,
  max_height: float = 3.0,
) -> torch.Tensor:
  robot: Entity = env.scene[entity_name]
  delta = robot.data.root_link_pos_w - env.scene.env_origins
  xy_error = torch.norm(delta[:, :2], dim=-1)
  height = robot.data.root_link_pos_w[:, 2]
  return (xy_error > max_xy_error) | (height < min_height) | (height > max_height)
