from mjdrone.tasks.hover import make_hover_env_cfg as make_hover_env_cfg
from mjdrone.tasks.hover import make_hover_runner_cfg as make_hover_runner_cfg
from mjdrone.tasks.waypoint import make_waypoint_env_cfg as make_waypoint_env_cfg
from mjdrone.tasks.waypoint import make_waypoint_runner_cfg as make_waypoint_runner_cfg

TASKS = {
  "hover": (make_hover_env_cfg, make_hover_runner_cfg),
  "waypoint": (make_waypoint_env_cfg, make_waypoint_runner_cfg),
}


def list_tasks() -> tuple[str, ...]:
  return tuple(TASKS.keys())


def make_env_cfg(
  task: str,
  *,
  play: bool,
  num_envs: int,
  image_height: int,
  image_width: int,
):
  return TASKS[task][0](
    play=play,
    num_envs=num_envs,
    image_height=image_height,
    image_width=image_width,
  )


def make_runner_cfg(task: str):
  return TASKS[task][1]()
