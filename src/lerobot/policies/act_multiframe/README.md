# ACT Multi-Frame Policy

An extension of the Action Chunking Transformer (ACT) policy that processes multiple adjacent frames as input, providing temporal context for improved action prediction.

## Overview

The ACT Multi-Frame policy extends the original ACT architecture to leverage temporal information from multiple consecutive observation frames. Instead of processing a single frame at each timestep, this policy processes a sequence of `n_obs_steps` frames, allowing it to capture motion and dynamics that are crucial for many robotic manipulation tasks.

## Key Features

- **Multi-Frame Input**: Process 2-10 adjacent frames simultaneously (configurable via `n_obs_steps`)
- **Temporal Positional Encoding**: Two encoding options:
  - `learned`: Learnable embeddings for each timestep
  - `sinusoidal`: Fixed sinusoidal positional encoding (as in "Attention Is All You Need")
- **Backward Compatible**: Can work with single frames (`n_obs_steps=1`) like standard ACT
- **Same Architecture**: Maintains the transformer encoder-decoder structure of ACT with optional VAE

## Architecture

```
Multi-Frame Processing Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Multiple Frames
  ├─ Frame t-2: (B, C, H, W)
  ├─ Frame t-1: (B, C, H, W)
  └─ Frame t:   (B, C, H, W)
           │
           ▼
  ┌────────────────────┐
  │ ResNet Backbone    │ ← Process each frame independently
  │ (shared weights)   │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Spatial Pos Enc    │ ← 2D positional encoding (H×W)
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Temporal Pos Enc   │ ← 1D positional encoding (timestep)
  └────────────────────┘
           │
           ▼
  Tokens: [t-2 spatial tokens] + [t-1 spatial tokens] + [t spatial tokens]
           │
           ▼
  ┌────────────────────┐
  │ Transformer        │
  │ Encoder + Decoder  │
  └────────────────────┘
           │
           ▼
     Action Chunks
```

## Usage

### Basic Example

```python
from lerobot.policies.act_multiframe import ACTMultiFrameConfig, ACTMultiFramePolicy
from lerobot.configs.types import Feature, FeatureType

# Create configuration with 3 observation steps
config = ACTMultiFrameConfig(
    n_obs_steps=3,  # Use 3 consecutive frames
    chunk_size=100,
    n_action_steps=100,
    temporal_encoding_type="learned",  # or "sinusoidal"
    input_features={
        "observation.images.cam_front": Feature(
            shape=[3, 3, 96, 96],  # (n_obs_steps, C, H, W)
            dtype="float32",
            name="observation.images.cam_front",
            feature_type=FeatureType.VISUAL,
        ),
        "observation.state": Feature(
            shape=[14],
            dtype="float32",
            name="observation.state",
            feature_type=FeatureType.STATE,
        ),
    },
    output_features={
        "action": Feature(
            shape=[14],
            dtype="float32",
            name="action",
            feature_type=FeatureType.ACTION,
        ),
    },
)

# Instantiate the policy
policy = ACTMultiFramePolicy(config)

# Prepare input batch with temporal dimension
batch = {
    "observation.images.cam_front": torch.randn(2, 3, 3, 96, 96),  # (B, T, C, H, W)
    "observation.state": torch.randn(2, 14),  # (B, state_dim)
}

# Inference
actions = policy.predict_action_chunk(batch)  # (B, chunk_size, action_dim)
```

### Training Example

```python
# Training mode requires actions
batch = {
    "observation.images.cam_front": torch.randn(32, 3, 3, 96, 96),
    "observation.state": torch.randn(32, 14),
    "action": torch.randn(32, 100, 14),
    "action_is_pad": torch.zeros(32, 100, dtype=torch.bool),
}

policy.train()
loss, loss_dict = policy.forward(batch)
```

## Configuration Parameters

Key parameters specific to multi-frame processing:

- **`n_obs_steps`** (int, default=3): Number of consecutive frames to process
  - Range: 1-10 (1 is equivalent to standard ACT)
  - Higher values provide more temporal context but increase computation

- **`temporal_encoding_type`** (str, default="learned"): Type of temporal encoding
  - `"learned"`: Trainable embeddings (recommended)
  - `"sinusoidal"`: Fixed sinusoidal positional encoding

All other parameters are inherited from standard ACT:
- `chunk_size`: Action prediction horizon (default: 100)
- `n_action_steps`: Actions executed per inference call (default: 100)
- `vision_backbone`: ResNet variant for image encoding (default: "resnet18")
- `use_vae`: Enable variational objective (default: True)
- `dim_model`: Transformer hidden dimension (default: 512)
- `n_encoder_layers`: Number of encoder layers (default: 4)
- `n_decoder_layers`: Number of decoder layers (default: 1)

## Data Format

### Multi-Frame Images

Images should have shape `(batch_size, n_obs_steps, channels, height, width)`:

```python
# Example with 3 frames
images = {
    "observation.images.cam_front": torch.randn(B, 3, 3, 96, 96),
    "observation.images.cam_wrist": torch.randn(B, 3, 3, 96, 96),
}
```

### Single-Frame Compatibility

The policy also accepts single frames `(B, C, H, W)` and automatically handles them:

```python
# Automatically expanded to (B, 1, C, H, W) internally
images = {
    "observation.images.cam_front": torch.randn(B, 3, 96, 96),
}
```

## Computational Considerations

- **Memory**: Increases linearly with `n_obs_steps`
  - 3 frames ≈ 3× memory of standard ACT for visual features

- **Sequence Length**: Each frame contributes `H × W` tokens
  - With ResNet18: typically 12×12 = 144 tokens per frame
  - Total tokens: `n_cameras × n_obs_steps × 144 + state_tokens`

- **Training Time**: Approximately `n_obs_steps`× slower than standard ACT

## When to Use Multi-Frame

✅ **Recommended for:**
- Tasks requiring velocity estimation (e.g., catching, tracking)
- Dynamic environments with moving objects
- Tasks where temporal patterns are important
- Scenarios with motion blur or uncertainty in single frames

❌ **May not help for:**
- Static pick-and-place tasks
- Environments with very slow dynamics
- Tasks where a single frame contains sufficient information
- Limited compute budgets

## Differences from Standard ACT

1. **Input Shape**: Images have temporal dimension `(B, T, C, H, W)` instead of `(B, C, H, W)`
2. **Temporal Encoding**: Additional positional encoding for timesteps
3. **Token Count**: `n_obs_steps`× more visual tokens in the transformer
4. **Configuration**: Removed validation that blocks `n_obs_steps > 1`

## References

Original ACT Paper:
```
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

## License

Same as the LeRobot project (Apache 2.0).
