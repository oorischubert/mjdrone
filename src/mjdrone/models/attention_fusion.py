"""Attention-based multimodal control model for image and low-dimensional observations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN
from tensordict import TensorDict


def _make_activation(name: str) -> nn.Module:
  name = name.lower()
  if name == "elu":
    return nn.ELU()
  if name == "relu":
    return nn.ReLU()
  if name == "gelu":
    return nn.GELU()
  if name == "silu":
    return nn.SiLU()
  if name == "tanh":
    return nn.Tanh()
  raise ValueError(f"Unsupported activation: {name}")


class PretrainedResNetBackbone(nn.Module):
  """Torchvision ResNet feature extractor that preserves spatial features."""

  _OUTPUT_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 512}

  def __init__(
    self,
    *,
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    output_stage: int = 3,
    trainable_stages: tuple[str, ...] | list[str] = ("layer3",),
    imagenet_norm: bool = True,
  ) -> None:
    super().__init__()
    try:
      from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
    except ImportError as exc:
      raise ImportError(
        "torchvision is required for pretrained vision backbones. "
        "Install project dependencies and retry."
      ) from exc

    if backbone_name == "resnet18":
      weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
      builder = resnet18
    elif backbone_name == "resnet34":
      weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
      builder = resnet34
    else:
      raise ValueError(f"Unsupported pretrained backbone: {backbone_name}")

    try:
      model = builder(weights=weights)
    except Exception as exc:
      raise RuntimeError(
        "Failed to load torchvision pretrained weights. "
        "If this is the first run, ensure internet access is available so "
        "torchvision can download the ImageNet checkpoint."
      ) from exc

    if output_stage not in self._OUTPUT_CHANNELS:
      raise ValueError(f"Unsupported output stage: {output_stage}. Expected 1-4.")

    self.output_stage = output_stage
    self.output_channels = self._OUTPUT_CHANNELS[output_stage]
    self.imagenet_norm = imagenet_norm

    self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    self.register_buffer("imagenet_mean", mean, persistent=False)
    self.register_buffer("imagenet_std", std, persistent=False)

    self._set_trainable_stages(tuple(trainable_stages))

  def _set_trainable_stages(self, trainable_stages: tuple[str, ...]) -> None:
    for parameter in self.parameters():
      parameter.requires_grad = False

    if not trainable_stages:
      return

    valid_stage_names = {"stem", "layer1", "layer2", "layer3", "layer4"}
    unknown = set(trainable_stages) - valid_stage_names
    if unknown:
      raise ValueError(f"Unknown trainable stages: {sorted(unknown)}")

    for stage_name in trainable_stages:
      stage = getattr(self, stage_name)
      for parameter in stage.parameters():
        parameter.requires_grad = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.imagenet_norm:
      x = (x - self.imagenet_mean) / self.imagenet_std

    x = self.stem(x)
    x = self.layer1(x)
    if self.output_stage == 1:
      return x
    x = self.layer2(x)
    if self.output_stage == 2:
      return x
    x = self.layer3(x)
    if self.output_stage == 3:
      return x
    return self.layer4(x)


class VisionAttentionModel(MLPModel):
  """Multimodal control model with state-conditioned visual attention."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, dict] | dict[str, Any],
    cnns: nn.ModuleDict | None = None,
    hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # noqa: B006
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)

    if not all(isinstance(v, dict) for v in cnn_cfg.values()):
      cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
    assert len(cnn_cfg) == len(self.obs_groups_2d), (
      "The number of CNN configurations must match the number of 2D observation groups."
    )

    sample_cfg = dict(cnn_cfg[self.obs_groups_2d[0]])
    self.attention_dim = int(sample_cfg.pop("attention_dim", 96))
    self.num_heads = int(sample_cfg.pop("attention_heads", 8))
    self.attention_dropout = float(sample_cfg.pop("attention_dropout", 0.0))
    self.num_query_tokens = int(sample_cfg.pop("num_query_tokens", 4))
    self.state_hidden_dim = int(sample_cfg.pop("state_hidden_dim", 64))
    self.state_latent_dim = int(sample_cfg.pop("state_latent_dim", self.attention_dim))
    self.include_pooled_visual = bool(sample_cfg.pop("include_pooled_visual", False))
    self.positional_hidden_dim = int(sample_cfg.pop("positional_hidden_dim", 32))
    state_scales = sample_cfg.pop("state_scales", None)
    pretrained_backbone_name = sample_cfg.pop("backbone_name", None)
    pretrained_backbone = bool(sample_cfg.pop("backbone_pretrained", False))
    backbone_output_stage = int(sample_cfg.pop("backbone_output_stage", 3))
    backbone_trainable_stages = sample_cfg.pop("backbone_trainable_stages", ("layer3",))
    imagenet_norm = bool(sample_cfg.pop("imagenet_norm", True))

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      _cnns = cnns
    else:
      _cnns = {}
      for idx, obs_group in enumerate(self.obs_groups_2d):
        group_cfg = dict(cnn_cfg[obs_group])
        for key in (
          "attention_dim",
          "attention_heads",
          "attention_dropout",
          "num_query_tokens",
          "state_hidden_dim",
          "state_latent_dim",
          "include_pooled_visual",
          "state_scales",
          "positional_hidden_dim",
          "spatial_softmax",
          "spatial_softmax_temperature",
          "global_pool",
          "flatten",
          "backbone_name",
          "backbone_pretrained",
          "backbone_output_stage",
          "backbone_trainable_stages",
          "imagenet_norm",
        ):
          group_cfg.pop(key, None)
        if pretrained_backbone_name is not None:
          if self.obs_channels_2d[idx] != 3:
            raise ValueError(
              "Pretrained torchvision backbones currently require 3-channel image inputs."
            )
          _cnns[obs_group] = PretrainedResNetBackbone(
            backbone_name=pretrained_backbone_name,
            pretrained=pretrained_backbone,
            output_stage=backbone_output_stage,
            trainable_stages=backbone_trainable_stages,
            imagenet_norm=imagenet_norm,
          )
        else:
          _cnns[obs_group] = CNN(
            input_dim=self.obs_dims_2d[idx],
            input_channels=self.obs_channels_2d[idx],
            global_pool="none",
            flatten=False,
            **group_cfg,
          )

    MLPModel.__init__(
      self,
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    act = _make_activation(activation)

    if state_scales is None:
      state_scales = [1.0] * self.obs_dim
    if len(state_scales) != self.obs_dim:
      raise ValueError(
        f"state_scales length {len(state_scales)} does not match obs_dim {self.obs_dim}."
      )
    self.register_buffer(
      "state_scales",
      torch.tensor(state_scales, dtype=torch.float32),
      persistent=False,
    )

    self.cnns = _cnns if isinstance(_cnns, nn.ModuleDict) else nn.ModuleDict(_cnns)
    self.visual_projections = nn.ModuleDict(
      {
        obs_group: nn.Linear(int(self.cnns[obs_group].output_channels), self.attention_dim)
        for obs_group in self.obs_groups_2d
      }
    )
    self.position_projection = nn.Sequential(
      nn.Linear(2, self.positional_hidden_dim),
      _make_activation(activation),
      nn.Linear(self.positional_hidden_dim, self.attention_dim),
    )

    self.state_encoder = nn.Sequential(
      nn.Linear(self.obs_dim, self.state_hidden_dim),
      nn.LayerNorm(self.state_hidden_dim),
      _make_activation(activation),
      nn.Linear(self.state_hidden_dim, self.state_latent_dim),
      nn.LayerNorm(self.state_latent_dim),
      _make_activation(activation),
    )
    self.state_to_queries = nn.Sequential(
      nn.Linear(self.state_latent_dim, self.state_latent_dim),
      nn.LayerNorm(self.state_latent_dim),
      act,
      nn.Linear(self.state_latent_dim, self.num_query_tokens * self.attention_dim),
    )
    self.cross_attention = nn.MultiheadAttention(
      embed_dim=self.attention_dim,
      num_heads=self.num_heads,
      dropout=self.attention_dropout,
      batch_first=True,
    )
    self.query_fusion = nn.Sequential(
      nn.Linear(self.num_query_tokens * self.attention_dim, self.attention_dim),
      nn.LayerNorm(self.attention_dim),
      _make_activation(activation),
    )
    fusion_input_dim = self.attention_dim * 2 + self.state_latent_dim
    self.fusion_gate = nn.Sequential(
      nn.Linear(fusion_input_dim, self.attention_dim),
      nn.LayerNorm(self.attention_dim),
      _make_activation(activation),
      nn.Linear(self.attention_dim, self.attention_dim),
      nn.Sigmoid(),
    )
    self.fusion_refine = nn.Sequential(
      nn.Linear(fusion_input_dim, self.attention_dim),
      nn.LayerNorm(self.attention_dim),
      _make_activation(activation),
      nn.Linear(self.attention_dim, self.attention_dim),
    )
    self.fusion_output = nn.Sequential(
      nn.LayerNorm(self.attention_dim),
      _make_activation(activation),
    )

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    del masks, hidden_state
    raw_state = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
    normalized_state = self.obs_normalizer(raw_state / self.state_scales)
    state_latent = self.state_encoder(normalized_state)

    image_tokens = []
    for obs_group in self.obs_groups_2d:
      feature_map = self.cnns[obs_group](obs[obs_group])
      batch, channels, height, width = feature_map.shape
      tokens = feature_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
      coords = self._get_position_tokens(height, width, feature_map.device, feature_map.dtype)
      projected_tokens = self.visual_projections[obs_group](tokens)
      image_tokens.append(projected_tokens + coords.expand(batch, -1, -1))
    visual_tokens = torch.cat(image_tokens, dim=1)

    query_tokens = self.state_to_queries(state_latent).view(
      state_latent.shape[0], self.num_query_tokens, self.attention_dim
    )
    attended_tokens, _ = self.cross_attention(
      query=query_tokens,
      key=visual_tokens,
      value=visual_tokens,
      need_weights=False,
    )
    attended_visual = self.query_fusion(attended_tokens.reshape(attended_tokens.shape[0], -1))
    pooled_visual = visual_tokens.mean(dim=1)

    fusion_input = torch.cat([attended_visual, pooled_visual, state_latent], dim=-1)
    gate = self.fusion_gate(fusion_input)
    refine = self.fusion_refine(fusion_input)
    fused_visual = self.fusion_output(
      gate * attended_visual + (1.0 - gate) * pooled_visual + refine
    )

    if self.include_pooled_visual:
      return torch.cat([state_latent, fused_visual, pooled_visual], dim=-1)
    return torch.cat([state_latent, fused_visual], dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      mlp_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
      self.obs_normalizer.update(mlp_obs / self.state_scales)  # type: ignore[arg-type]

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    obs_dim_1d = 0
    obs_groups_1d = []
    obs_dims_2d = []
    obs_channels_2d = []
    obs_groups_2d = []

    for obs_group in active_obs_groups:
      if len(obs[obs_group].shape) == 4:
        obs_groups_2d.append(obs_group)
        obs_dims_2d.append(obs[obs_group].shape[2:4])
        obs_channels_2d.append(obs[obs_group].shape[1])
      elif len(obs[obs_group].shape) == 2:
        obs_groups_1d.append(obs_group)
        obs_dim_1d += obs[obs_group].shape[-1]
      else:
        raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

    assert obs_groups_2d, (
      "No 2D observations are provided. Use an MLP-only model if the task has no images."
    )

    self.obs_dims_2d = obs_dims_2d
    self.obs_channels_2d = obs_channels_2d
    self.obs_groups_2d = obs_groups_2d
    return obs_groups_1d, obs_dim_1d

  def _get_latent_dim(self) -> int:
    latent_dim = self.state_latent_dim + self.attention_dim
    if self.include_pooled_visual:
      latent_dim += self.attention_dim
    return latent_dim

  def _get_position_tokens(
    self,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
  ) -> torch.Tensor:
    y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
    return self.position_projection(coords)
