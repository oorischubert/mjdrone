# mjdrone

`mjdrone` is a standalone multimodal drone RL project built on top of the sibling [`mjlab`](../mjlab) repository.

The project now has three defining characteristics:

- a **shared attention-fusion policy backbone** that combines front-camera vision with IMU and task-state inputs,
- **randomized visual environments** so the policy learns across many layouts instead of memorizing one scene,
- a **staged curriculum** that starts with hover, moves to waypoint tracking, and is intended to extend to gate flight.

The current vision backbone is no longer trained from scratch. The policy now uses a **pretrained ResNet image encoder** with selective fine-tuning, then fuses those visual features with IMU/state inputs through the same attention-fusion control head.

Today the project contains two tasks:

- `hover`: lift off from the ground and stabilize near a low hover target,
- `waypoint`: lift off from the ground and intercept a distinct target tank while avoiding all non-target contacts.

The near-term goal is not gate flight yet. The goal is to build the shared vehicle, sensing, training, and policy stack that gate flight will later depend on.

There is also a play-only inspection mode:

- `test`: loads the shared environment for viewing and disables rotor thrust so the drone stays inert on the ground.

## Project Layout

```text
mjdrone/
├── README.md
├── pyproject.toml
└── src/
    └── mjdrone/
        ├── cli.py
        ├── assets/
        │   ├── __init__.py
        │   ├── landmarks.py
        │   └── quadcopter.py
        ├── models/
        │   ├── __init__.py
        │   └── attention_fusion.py
        └── tasks/
            ├── __init__.py
            ├── hover/
            │   ├── __init__.py
            │   ├── env_cfg.py
            │   ├── mdp.py
            │   └── rl_cfg.py
            └── waypoint/
                ├── __init__.py
                ├── env_cfg.py
                ├── mdp.py
                └── rl_cfg.py
```

## Shared System Components

These parts are shared across tasks and are the real core of the project.

### Vehicle And Sensors

The quadcopter asset is defined in [`src/mjdrone/assets/quadcopter.py`](./src/mjdrone/assets/quadcopter.py).

It includes:

- a floating-base quadcopter body,
- four rotor thrust actuators,
- an `imu_site` used for acceleration, velocity, and gyro sensing,
- a front-facing RGB camera,
- visual geometry that is hidden from the onboard camera so the camera does not see the drone itself.

The default onboard camera now uses a landscape image shape of `160 x 96`, a corrected roll orientation, and a narrower field of view than before, so target vehicles occupy more useful pixels and the feed displays with the expected horizon orientation.

The default action space is 4D, one normalized thrust command per rotor.

### Observation Structure

Each task builds three observation groups:

- `actor`
- `critic`
- `camera`

`actor` and `critic` contain the low-dimensional state:

- IMU linear acceleration,
- IMU linear velocity,
- IMU angular velocity,
- projected gravity,
- target error terms,
- previous action.

`camera` contains the front RGB image.

The runner config resolves observations as:

- `actor = ("actor", "camera")`
- `critic = ("critic", "camera")`

### Policy Backbone

Both hover and waypoint use the same multimodal attention-fusion backbone implemented in [`src/mjdrone/models/attention_fusion.py`](./src/mjdrone/models/attention_fusion.py).

The model is intentionally split into a fast control/state path and a visual context path:

- the **state path** handles IMU and task state,
- the **vision path** handles the camera image through a pretrained ResNet backbone,
- the **fusion block** lets state decide what visual evidence matters.

#### Why Not YOLO

Using YOLO directly for this project would be the wrong abstraction.

YOLO is an object detector. It is useful when you already have bounding-box supervision and want explicit detections. This project is an end-to-end RL control problem, and the policy needs dense scene features for stabilization, obstacle avoidance, and target pursuit, not just detector boxes.

So instead of putting a detector in the loop, the project now uses a pretrained image backbone and fine-tunes it for control. That gives you transferable visual features without forcing a separate detection dataset or hard-wiring the policy around detector outputs.

LoRA-style adaptation is possible in principle, but for this codebase the higher-value first step is selective fine-tuning of a pretrained backbone. It is simpler, better aligned with the current model, and much easier to maintain.

#### Pretrained Vision Encoder

The vision branch now uses:

- `torchvision` ResNet-18 pretrained on ImageNet,
- ImageNet input normalization,
- feature extraction up to `layer3`,
- only the later visual stage (`layer3`) left trainable by default,
- shared visual encoder weights between actor and critic during PPO.

This gives the policy a stronger starting point than a randomly initialized CNN while keeping the multimodal attention and control head task-specific.

#### Why The IMU Matters

The IMU is not a side input. It is part of the main control pathway.

The low-dimensional state branch contains:

- linear acceleration,
- linear velocity,
- angular velocity,
- projected gravity,
- target error,
- previous action.

That branch gives the policy direct information about motion, tilt, drift, and recent control effort. Vision provides scene context; the IMU/state branch provides fast stabilization context.

#### Model Structure

High-level data flow:

```text
camera image
  -> pretrained ResNet-18 feature extractor
  -> visual tokens + 2D positional encoding

normalized IMU/task state
  -> state encoder MLP
  -> encoded state latent
  -> 4 state-derived query tokens

state-derived query tokens + visual tokens
  -> cross-attention
  -> attended visual

encoded state latent + attended visual
  -> state-conditioned gated fusion
  -> fused latent
  -> actor head or critic head
```

Stage-by-stage view:

| Stage | Input | Operation | Output |
| --- | --- | --- | --- |
| Vision encoder | `3 x 96 x 160` RGB image | pretrained ResNet-18 up to `layer3` | visual feature map |
| Vision tokenization | feature map | flatten spatial dimensions | `N x 96` visual tokens |
| Positional encoding | token grid | learned 2D position projection | position-aware tokens |
| State encoder | 21D IMU/task vector | MLP `21 -> 64 -> 96` with normalization | 96D state latent |
| Query generation | 96D state latent | linear projection | `4 x 96` query tokens |
| Cross-attention | state queries + visual tokens | multi-head attention | attended visual summary |
| Fusion | state latent + attended visual + pooled visual context | gated fusion block | fused visual latent |
| Final latent | state latent + fused visual | concatenation | 192D policy/value latent |
| Output head | 192D latent | MLP `512 -> 384 -> 256` | actor actions or critic value |

Current default dimensions for both tasks:

- image input: `3 x 96 x 160`
- low-dimensional state input: `21`
- state encoder: `21 -> 64 -> 96`
- attention width: `96`
- attention heads: `8`
- query tokens: `4`
- final fused latent: `192`
- actor hidden layers: `512 -> 384 -> 256 -> 4`
- critic hidden layers: `512 -> 384 -> 256 -> 1`

#### Weight Download Requirement

If the pretrained ResNet weights are not already cached on your machine, the first training or play run that constructs the model will trigger a `torchvision` weight download.

That means:

- `torchvision` must be installed,
- the machine must have internet access for the first pretrained run,
- after the weights are cached locally, later runs reuse them.

Design intent:

- the state branch stays separate until late fusion,
- state-derived queries attend over image tokens, not the reverse,
- the encoded state has a direct path to the output head,
- the policy can stabilize from IMU/state even when vision is noisy,
- vision provides external context for drift correction, orientation references, and later target tracking.

### Visual Scene Randomization

The current project includes **visual scene randomization**, not full dynamics/domain randomization.

That distinction matters.

Current visual randomization changes the rendered scene so the camera policy must generalize across layouts. It does **not** yet change the underlying flight dynamics.

The reusable landmark assets are defined in [`src/mjdrone/assets/landmarks.py`](./src/mjdrone/assets/landmarks.py). On environment reset, task code randomizes:

- a multi-block branching road network around the drone,
- ground and road-like coloring,
- tree positions, yaw, and canopy colors,
- car positions, yaw, and body colors,
- billboard positions, yaw, and panel colors.

This makes different parallel environments look different and causes each episode to present a new scene.

Future dynamics/domain randomization is still separate work. That future stage would include things like:

- wind disturbances,
- mass or inertia variation,
- motor lag,
- sensor noise tuning,
- sim-to-real parameter randomization.

## Tasks

### Hover

Hover is the stabilization task and the base of the curriculum.

The hover task:

- starts with the drone on the ground instead of already airborne,
- samples a local target inside a small 3D region,
- keeps that target low enough to require liftoff without pushing the drone above the scene,
- rewards planar position and altitude tracking,
- rewards staying upright,
- penalizes post-liftoff contact so crashing is explicitly bad instead of only terminating the episode,
- penalizes linear velocity, angular velocity, action rate, and action magnitude,
- terminates on time-out, excessive tilt, post-liftoff contact, or leaving the allowed flight region.

This task is designed to teach basic visual-inertial stabilization before navigation is added.

Main files:

- [`src/mjdrone/tasks/hover/env_cfg.py`](./src/mjdrone/tasks/hover/env_cfg.py)
- [`src/mjdrone/tasks/hover/mdp.py`](./src/mjdrone/tasks/hover/mdp.py)
- [`src/mjdrone/tasks/hover/rl_cfg.py`](./src/mjdrone/tasks/hover/rl_cfg.py)

### Waypoint

Waypoint builds on the same vehicle, sensors, randomization, and policy backbone as hover.

What changes relative to hover:

- the drone also starts on the ground and must lift off before navigating,
- the target is a single tan tank-like vehicle that is visually distinct from the ordinary cars in the scene,
- the actor no longer receives the true target offset directly and instead must use the camera to identify and pursue the target tank,
- the tank is spawned anywhere in the environment and the drone is initialized facing it so it starts in frame,
- the reward emphasizes progress and fast interception of the target vehicle,
- successful target contact gets a large positive bonus,
- non-target post-liftoff contact gets an explicit negative penalty,
- colliding with the target tank is success,
- any post-liftoff contact with the ground or other obstacles is failure.

So hover teaches the drone to lift off and remain stable; waypoint teaches it to lift off, visually acquire the target tank, and move through clutter without touching anything else.

Main files:

- [`src/mjdrone/tasks/waypoint/env_cfg.py`](./src/mjdrone/tasks/waypoint/env_cfg.py)
- [`src/mjdrone/tasks/waypoint/mdp.py`](./src/mjdrone/tasks/waypoint/mdp.py)
- [`src/mjdrone/tasks/waypoint/rl_cfg.py`](./src/mjdrone/tasks/waypoint/rl_cfg.py)

## Training And Checkpoints

Training uses:

- `mjlab` environment managers,
- `RslRlVecEnvWrapper`,
- PPO through `MjlabOnPolicyRunner`.

Runs are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>[_run_name]/
```

Each run stores:

- checkpoints,
- `params/env.yaml`,
- `params/agent.yaml`,
- optional training videos when `--video` is enabled.

### Where The Trained Model Is Saved

Model checkpoints are saved as:

```text
logs/rsl_rl/<experiment_name>/<timestamp>[_run_name]/model_<iteration>.pt
```

Current experiment directories by task:

- `hover`: `logs/rsl_rl/mjdrone_hover_pretrained_vision/`
- `waypoint`: `logs/rsl_rl/mjdrone_waypoint_pretrained_vision/`

Examples:

```text
/home/oorischubert/mjdrone/logs/rsl_rl/mjdrone_hover_pretrained_vision/<timestamp>/model_<iteration>.pt
/home/oorischubert/mjdrone/logs/rsl_rl/mjdrone_waypoint_pretrained_vision/<timestamp>/model_<iteration>.pt
```

If you want the latest checkpoint for a task, look inside the newest timestamped run directory under that experiment.

The currently retained waypoint run is:

```text
/home/oorischubert/mjdrone/logs/rsl_rl/mjdrone_waypoint_pretrained_vision/2026-03-11_13-02-32/model_3999.pt
```

Useful commands:

```bash
ls /home/oorischubert/mjdrone/logs/rsl_rl
ls /home/oorischubert/mjdrone/logs/rsl_rl/mjdrone_hover_pretrained_vision
ls /home/oorischubert/mjdrone/logs/rsl_rl/mjdrone_waypoint_pretrained_vision
find /home/oorischubert/mjdrone/logs/rsl_rl -name 'model_*.pt' | sort
```

## Headless And Viewer Modes

Interactive playback is controlled through `mjdrone play`.

Default behavior:

- `--headless` defaults to **false**,
- if `--headless` is omitted, playback opens a viewer,
- if `--headless` is passed, playback runs without a viewer.

Viewer selection:

- `--viewer auto`: prefer `viser` for camera-based tasks,
- `--viewer native`: force the MuJoCo native viewer,
- `--viewer viser`: force the browser-based Viser viewer.

For camera-based inspection, `viser` is the better option because it shows the onboard camera feed in `Camera Feeds`. The native viewer only shows the external scene.

## Setup

Expected sibling layout:

```text
/home/oorischubert/
├── mjlab/
└── mjdrone/
```

From `mjdrone/`:

```bash
uv sync
```

`pyproject.toml` points to `../mjlab` as an editable dependency.

## Commands

### Train

```bash
uv run mjdrone train
```

Train hover explicitly:

```bash
uv run mjdrone train --task hover
```

Train waypoint explicitly:

```bash
uv run mjdrone train --task waypoint
```

Useful training flags:

- `--task hover|waypoint`
- `--device cuda:0`
- `--num-envs 512`
- `--seed 42`
- `--max-iterations N`
- `--num-steps-per-env N`
- `--run-name name`
- `--log-root /custom/path`
- `--image-width 64`
- `--image-height 48`
- `--video`
- `--video-length 200`
- `--video-interval 2000`

If `--max-iterations` or `--num-steps-per-env` are omitted, task defaults from the runner config are used:

- `hover`: `3000` iterations, `32` steps per environment
- `waypoint`: `4000` iterations, `40` steps per environment

Example:

```bash
uv run mjdrone train   --task hover   --device cuda:0   --num-envs 512   --run-name hover_rgb_imu   --video
```

### Play

Because `--headless` defaults to false, this opens a viewer:

```bash
uv run mjdrone play
```

Play waypoint:

```bash
uv run mjdrone play --task waypoint
```

In waypoint play, the environment spawns one visually distinct tan target tank at reset. The drone should lift off, find that vehicle in the camera feed, and collide with it without touching anything else first.

Show the onboard camera feed explicitly:

```bash
uv run mjdrone play --viewer viser
```

Useful playback flags:

- `--task hover|waypoint|test`
- `--agent trained|random|zero`
- `--checkpoint /path/to/model.pt`
- `--device cuda:0`
- `--num-envs 1`
- `--viewer auto|native|viser`
- `--headless`
- `--steps 1000`
- `--video`
- `--video-length 300`
- `--image-width 160`
- `--image-height 96`

Examples:

```bash
uv run mjdrone play --agent random
uv run mjdrone play --viewer native
uv run mjdrone play --viewer viser
uv run mjdrone play --task test
uv run mjdrone play --headless --steps 1500
```

To save a video during headless playback:

```bash
uv run mjdrone play --headless --video --video-length 300
```

## Why The Trees, Cars, And Billboards Exist

The trees, parked cars, billboards, and ground markings serve two roles:

- they provide visual structure for the front camera,
- and in waypoint they also act as physical obstacles that the drone must avoid after liftoff.

They exist because camera-based hover and waypoint learning are poorly conditioned if the image mostly contains blank ground and sky. The landmarks provide:

- stable texture,
- depth and parallax cues,
- orientation references,
- better visual evidence for drift and yaw correction.

With reset-time randomization, they are also part of the generalization strategy rather than fixed decoration. The target tank is randomized in the same shared scene generator, but it is kept in a forward spawn corridor so the drone can see it immediately at the start of each episode.

## Development Sequence

The intended progression is:

1. Hover
2. Waypoint tracking
3. Single gate flight
4. Multi-gate racing

That order is deliberate. Gate flight should build on a stable hover and navigation stack, not replace it.

## Current Limitations

This is still an early project.

It does **not** yet include:

- gate geometry or gate-passage rewards,
- ordered waypoint sequences beyond single active target vehicles,
- dynamics/domain randomization such as wind or motor lag,
- sim-to-real tuning,
- multi-camera perception,
- higher-level controller abstractions above raw rotor thrust,
- temporal memory for deliberate target-search behavior when the target leaves the camera frustum.
