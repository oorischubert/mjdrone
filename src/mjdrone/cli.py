"""Command line entrypoints for training and evaluating the hover task."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from mjdrone.tasks import list_tasks, make_env_cfg, make_runner_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import CameraSensor
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAY_TASKS = list_tasks() + ("test",)


def _resolve_device(device: str | None) -> str:
  if device:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _timestamped_run_dir(root: Path, run_name: str) -> Path:
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if run_name:
    log_dir_name += f"_{run_name}"
  return root / log_dir_name


def _build_policy(
  task: str,
  agent: str,
  env: RslRlVecEnvWrapper,
  checkpoint: Path | None,
  device: str,
) -> Callable:
  if agent == "zero":
    action_shape = tuple(env.unwrapped.action_space.shape)

    class PolicyZero:
      def __call__(self, obs):
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    return PolicyZero()

  if agent == "random":
    action_shape = tuple(env.unwrapped.action_space.shape)

    class PolicyRandom:
      def __call__(self, obs):
        del obs
        return 2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0

    return PolicyRandom()

  if checkpoint is None:
    raise ValueError("A checkpoint is required when agent='trained'.")

  runner_cfg = make_runner_cfg(task)
  runner = MjlabOnPolicyRunner(env, asdict(runner_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def _default_log_root(experiment_name: str) -> Path:
  return PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name


def _resolve_latest_checkpoint(log_root: Path) -> Path:
  return get_checkpoint_path(log_root, checkpoint=r"model_.*\.pt")


def _env_has_camera_sensors(env: ManagerBasedRlEnv) -> bool:
  return any(
    isinstance(sensor, CameraSensor) for sensor in env.scene.sensors.values()
  )


def train(args: argparse.Namespace) -> None:
  configure_torch_backends()
  device = _resolve_device(args.device)

  env_cfg = make_env_cfg(
    args.task,
    play=False,
    num_envs=args.num_envs,
    image_height=args.image_height,
    image_width=args.image_width,
  )
  runner_cfg = make_runner_cfg(args.task)

  env_cfg.seed = args.seed
  runner_cfg.seed = args.seed
  if args.max_iterations is not None:
    runner_cfg.max_iterations = args.max_iterations
  if args.num_steps_per_env is not None:
    runner_cfg.num_steps_per_env = args.num_steps_per_env
  runner_cfg.run_name = args.run_name

  log_root = Path(args.log_root) if args.log_root else _default_log_root(
    runner_cfg.experiment_name
  )
  log_dir = _timestamped_run_dir(log_root, args.run_name)

  render_mode = "rgb_array" if args.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  if args.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "train",
      step_trigger=lambda step: step % args.video_interval == 0,
      video_length=args.video_length,
      disable_logger=True,
    )

  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped_env, asdict(runner_cfg), str(log_dir), device)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(runner_cfg))

  print(f"[INFO] Training on {device}")
  print(f"[INFO] Logging to {log_dir}")
  runner.learn(
    num_learning_iterations=runner_cfg.max_iterations,
    init_at_random_ep_len=True,
  )
  wrapped_env.close()


def play(args: argparse.Namespace) -> None:
  configure_torch_backends()
  device = _resolve_device(args.device)
  task_name = "hover" if args.task == "test" else args.task
  agent_name = args.agent
  if args.task == "test" and args.agent == "trained" and args.checkpoint is None:
    agent_name = "zero"
    print("[INFO] Test mode uses the hover environment with rotor thrust disabled.")

  env_cfg = make_env_cfg(
    task_name,
    play=True,
    num_envs=args.num_envs,
    image_height=args.image_height,
    image_width=args.image_width,
  )
  if args.task == "test":
    env_cfg.actions["rotor_thrust"].scale = 0.0
    env_cfg.actions["rotor_thrust"].offset = 0.0
  runner_cfg = make_runner_cfg(task_name)

  checkpoint: Path | None = None
  if agent_name == "trained":
    if args.checkpoint:
      checkpoint = Path(args.checkpoint).expanduser().resolve()
      if not checkpoint.exists() or not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    else:
      checkpoint = _resolve_latest_checkpoint(_default_log_root(runner_cfg.experiment_name))
    print(f"[INFO] Loading checkpoint: {checkpoint}")

  render_mode = "rgb_array" if args.headless and args.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  if args.headless and args.video:
    video_root = checkpoint.parent if checkpoint is not None else PROJECT_ROOT / "logs"
    env = VideoRecorder(
      env,
      video_folder=video_root / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=args.video_length,
      disable_logger=True,
    )

  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
  policy = _build_policy(task_name, agent_name, wrapped_env, checkpoint, device)

  if args.headless:
    obs, _ = wrapped_env.reset()
    for _ in range(args.steps):
      actions = policy(obs)
      obs, _, _, _ = wrapped_env.step(actions)
    wrapped_env.close()
    return

  if args.viewer == "auto":
    has_display = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
    if has_display and not _env_has_camera_sensors(wrapped_env.unwrapped):
      resolved_viewer = "native"
    else:
      resolved_viewer = "viser"
  else:
    resolved_viewer = args.viewer

  if resolved_viewer == "viser" and _env_has_camera_sensors(wrapped_env.unwrapped):
    print(
      "[INFO] Using Viser so the drone camera feed is visible in the Camera Feeds panel."
    )

  if resolved_viewer == "native":
    NativeMujocoViewer(wrapped_env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(wrapped_env, policy).run()
  else:
    raise ValueError(f"Unsupported viewer: {resolved_viewer}")
  wrapped_env.close()


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="mjdrone task runner")
  subparsers = parser.add_subparsers(dest="command", required=True)

  train_parser = subparsers.add_parser("train", help="Train a policy.")
  train_parser.add_argument(
    "--task",
    choices=list_tasks(),
    default="hover",
    help="Task to train.",
  )
  train_parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
  train_parser.add_argument("--num-envs", type=int, default=512, help="Parallel env count.")
  train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
  train_parser.add_argument(
    "--max-iterations",
    type=int,
    default=None,
    help="Override the task default number of PPO updates.",
  )
  train_parser.add_argument(
    "--num-steps-per-env",
    type=int,
    default=None,
    help="Override the task default rollout length per PPO update.",
  )
  train_parser.add_argument("--run-name", default="", help="Optional run label.")
  train_parser.add_argument(
    "--log-root",
    default=None,
    help="Override log root. Default is ./logs/rsl_rl/<experiment_name>.",
  )
  train_parser.add_argument(
    "--image-height",
    type=int,
    default=96,
    help="Front camera image height.",
  )
  train_parser.add_argument(
    "--image-width",
    type=int,
    default=160,
    help="Front camera image width.",
  )
  train_parser.add_argument(
    "--video",
    action="store_true",
    help="Record periodic training videos.",
  )
  train_parser.add_argument(
    "--video-length",
    type=int,
    default=200,
    help="Frames per recorded training clip.",
  )
  train_parser.add_argument(
    "--video-interval",
    type=int,
    default=2000,
    help="Environment-step interval between training videos.",
  )
  train_parser.set_defaults(func=train)

  play_parser = subparsers.add_parser("play", help="Run a trained rollout.")
  play_parser.add_argument(
    "--task",
    choices=PLAY_TASKS,
    default="hover",
    help="Task to play.",
  )
  play_parser.add_argument(
    "--agent",
    choices=("trained", "zero", "random"),
    default="trained",
    help="Which policy to run.",
  )
  play_parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
  play_parser.add_argument("--num-envs", type=int, default=1, help="Parallel env count.")
  play_parser.add_argument(
    "--checkpoint",
    default=None,
    help="Checkpoint path. If omitted, the latest local checkpoint is used.",
  )
  play_parser.add_argument(
    "--steps",
    type=int,
    default=1000,
    help="Headless rollout length in environment steps.",
  )
  play_parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without an interactive viewer. Default is false.",
  )
  play_parser.add_argument(
    "--viewer",
    choices=("auto", "native", "viser"),
    default="auto",
    help="Viewer backend used when not headless.",
  )
  play_parser.add_argument(
    "--image-height",
    type=int,
    default=96,
    help="Front camera image height.",
  )
  play_parser.add_argument(
    "--image-width",
    type=int,
    default=160,
    help="Front camera image width.",
  )
  play_parser.add_argument(
    "--video",
    action="store_true",
    help="Record a video during headless playback.",
  )
  play_parser.add_argument(
    "--video-length",
    type=int,
    default=300,
    help="Frames to capture when --video is enabled.",
  )
  play_parser.set_defaults(func=play)

  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
