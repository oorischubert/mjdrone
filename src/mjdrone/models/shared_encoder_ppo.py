"""RL algorithm extension for shared visual encoders."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils import resolve_optimizer


def _unique_parameters(*parameter_groups: Iterable) -> list:
  unique: list = []
  seen_ids: set[int] = set()
  for parameter in chain(*parameter_groups):
    parameter_id = id(parameter)
    if parameter_id in seen_ids:
      continue
    seen_ids.add(parameter_id)
    unique.append(parameter)
  return unique


class SharedEncoderPPO(PPO):
  """PPO variant that deduplicates shared parameters before optimizer creation."""

  def __init__(self, *args, optimizer: str = "adam", learning_rate: float = 0.001, **kwargs):
    super().__init__(*args, optimizer=optimizer, learning_rate=learning_rate, **kwargs)
    unique_params = _unique_parameters(self.actor.parameters(), self.critic.parameters())
    self.optimizer = resolve_optimizer(optimizer)(unique_params, lr=learning_rate)  # type: ignore[arg-type]
