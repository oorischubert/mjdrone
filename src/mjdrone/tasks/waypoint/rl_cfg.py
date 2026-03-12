"""RL config for waypoint tracking."""

from mjdrone.tasks.hover.rl_cfg import make_hover_runner_cfg


def make_waypoint_runner_cfg():
  cfg = make_hover_runner_cfg()
  cfg.actor.cnn_cfg["state_scales"][16] = 1.0
  cfg.critic.cnn_cfg["state_scales"][16] = 1.0
  cfg.experiment_name = "mjdrone_waypoint_pretrained_vision"
  cfg.max_iterations = 4_000
  cfg.num_steps_per_env = 40
  cfg.save_interval = 100
  return cfg
