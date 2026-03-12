"""RL config for the hover task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def make_hover_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = {
    "backbone_name": "resnet18",
    "backbone_pretrained": True,
    "backbone_output_stage": 3,
    "backbone_trainable_stages": ("layer3",),
    "imagenet_norm": True,
    "attention_dim": 96,
    "attention_heads": 8,
    "attention_dropout": 0.0,
    "num_query_tokens": 4,
    "state_hidden_dim": 64,
    "state_latent_dim": 96,
    "include_pooled_visual": False,
    "positional_hidden_dim": 32,
    "state_scales": [
      15.0, 15.0, 15.0,
      5.0, 5.0, 5.0,
      8.0, 8.0, 8.0,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      1.0,
      3.14159265,
      1.0, 1.0, 1.0, 1.0,
    ],
  }
  class_name = "mjdrone.models.attention_fusion:VisionAttentionModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 384, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.7,
        "std_type": "log",
      },
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 384, 256),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=7.5e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      share_cnn_encoders=True,
      class_name="mjdrone.models.shared_encoder_ppo:SharedEncoderPPO",
    ),
    experiment_name="mjdrone_hover_pretrained_vision",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=3_000,
    clip_actions=1.0,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )
