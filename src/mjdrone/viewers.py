"""Custom viewers for interactive mjdrone tasks."""

from __future__ import annotations

from typing import Any

import torch

from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction


class WaypointClickViewer(ViserPlayViewer):
  """Viser viewer that lets users click to place waypoint targets."""

  def __init__(self, env, policy, frame_rate: float = 60.0, verbosity=0) -> None:
    super().__init__(env, policy, frame_rate=frame_rate, verbosity=verbosity)
    self._target_height_slider = None
    self._waypoint_status = None

  def setup(self) -> None:
    super().setup()

    with self._server.gui.add_folder("Waypoint Target"):
      self._waypoint_status = self._server.gui.add_html(
        "<div style='font-size:0.85em; line-height:1.35;'>"
        "Click the scene to place a waypoint.<br/>"
        "The click sets <strong>x/y</strong> on the ground plane and the slider sets the flight height."
        "</div>"
      )
      self._target_height_slider = self._server.gui.add_slider(
        "Target height",
        min=0.85,
        max=1.45,
        step=0.05,
        initial_value=1.05,
      )
      clear_button = self._server.gui.add_button("Clear Target")

      @clear_button.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", {"type": "clear_waypoint"})

    @self._server.scene.on_pointer_event("click")
    def _(event) -> None:
      target = self._target_from_click(event.ray_origin, event.ray_direction)
      if target is None:
        return
      self.request_action("CUSTOM", {"type": "set_waypoint", "target": target})

  def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
    if action != ViewerAction.CUSTOM or not isinstance(payload, dict):
      return False

    payload_type = payload.get("type")
    if payload_type == "clear_waypoint":
      self._clear_waypoint()
      return True
    if payload_type == "set_waypoint":
      target = payload.get("target")
      if target is None:
        return False
      self._set_waypoint(target)
      return True
    return False

  def _target_from_click(
    self,
    ray_origin: tuple[float, float, float] | None,
    ray_direction: tuple[float, float, float] | None,
  ) -> tuple[float, float, float] | None:
    if ray_origin is None or ray_direction is None:
      return None

    origin_z = float(ray_origin[2])
    direction_z = float(ray_direction[2])
    if abs(direction_z) < 1.0e-5:
      return None

    distance = -origin_z / direction_z
    if distance <= 0.0:
      return None

    hit_x = float(ray_origin[0] + distance * ray_direction[0])
    hit_y = float(ray_origin[1] + distance * ray_direction[1])
    target_height = 1.05 if self._target_height_slider is None else float(self._target_height_slider.value)
    return (hit_x, hit_y, target_height)

  def _waypoint_term(self):
    return self.env.unwrapped.command_manager.get_term("waypoint")

  def _selected_env_idx(self) -> int:
    return getattr(self._scene, "env_idx", 0)

  def _set_waypoint(self, target_world: tuple[float, float, float]) -> None:
    waypoint_term = self._waypoint_term()
    env_idx = self._selected_env_idx()
    env_ids = torch.tensor([env_idx], device=self.env.unwrapped.device, dtype=torch.long)
    origin = self.env.unwrapped.scene.env_origins[env_ids][0]

    clamped = torch.tensor(target_world, device=self.env.unwrapped.device, dtype=torch.float32)
    ranges = waypoint_term.cfg.target_position_range
    clamped[0] = torch.clamp(clamped[0] - origin[0], ranges.x[0], ranges.x[1]) + origin[0]
    clamped[1] = torch.clamp(clamped[1] - origin[1], ranges.y[0], ranges.y[1]) + origin[1]
    clamped[2] = torch.clamp(clamped[2], ranges.z[0], ranges.z[1])
    waypoint_term.set_target_world(clamped.unsqueeze(0), env_ids)
    self._needs_update = True

  def _clear_waypoint(self) -> None:
    waypoint_term = self._waypoint_term()
    env_idx = self._selected_env_idx()
    env_ids = torch.tensor([env_idx], device=self.env.unwrapped.device, dtype=torch.long)
    waypoint_term.clear_target(env_ids)
    self._needs_update = True
