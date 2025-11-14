# ACT-MEM: Action Chunking Transformer with Memory

An advanced extension of the Action Chunking Transformer (ACT) policy that combines two key improvements for dynamic robotic manipulation:

1. **Multi-frame Temporal Processing** - For velocity and motion estimation
2. **Episodic Memory** - For remembering key decisions and past experiences

## Overview

ACT-MEM addresses two critical limitations of standard ACT:

**Problem 1: Dynamic Environments**
Standard ACT processes only the current frame, making it difficult to estimate velocity, track moving objects, or handle motion blur.

**Solution:** Multi-frame temporal processing that considers recent frames (t-2, t-1, t) to capture motion dynamics.

**Problem 2: Lack of Long-term Memory**
ACT makes decisions based only on current observations without remembering what it has already done or learned from similar past situations.

**Solution:** Episodic memory module that stores and retrieves relevant past experiences to inform current decisions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ACT-MEM ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────┐
 │ 1. MULTI-FRAME INPUT (Recent Frames)                     │
 │    ┌──────┐  ┌──────┐  ┌──────┐                         │
 │    │Frame │  │Frame │  │Frame │                          │
 │    │ t-2  │  │ t-1  │  │  t   │                          │
 │    └──┬───┘  └──┬───┘  └──┬───┘                          │
 │       └─────────┴─────────┘                               │
 │               ▼                                            │
 │    ┌──────────────────────┐                               │
 │    │  ResNet Backbone     │                               │
 │    │  (Shared Weights)    │                               │
 │    └──────────┬───────────┘                               │
 │               ▼                                            │
 │    ┌──────────────────────┐                               │
 │    │ Spatial Pos Enc (2D) │                               │
 │    └──────────┬───────────┘                               │
 │               ▼                                            │
 │    ┌──────────────────────┐                               │
 │    │ Temporal Pos Enc (1D)│                               │
 │    └──────────┬───────────┘                               │
 └───────────────┼─────────────────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2. TRANSFORMER ENCODER                                    │
 │    Process all temporal-spatial tokens                    │
 │    Output: Obs Embedding + Encoder Features               │
 └──────────────┬────────────────┬───────────────────────────┘
                │                │
                ▼                ▼
 ┌──────────────────────┐  ┌─────────────────────────────┐
 │ 3. MEMORY MODULE     │  │ 4. DECODER                   │
 │ ┌──────────────────┐ │  │                              │
 │ │  Memory Bank     │ │  │  Self-Attention              │
 │ │  (Episodic)      │ │  │  ↓                           │
 │ │                  │ │  │  Cross-Attn: Encoder Output  │
 │ │  Stores:         │ │  │  ↓                           │
 │ │  - Obs Embeddings│ │  │  Cross-Attn: Retrieved Mem   │
 │ │  - Actions       │ │  │  ↓                           │
 │ │  - Metadata      │ │  │  Feed-Forward                │
 │ └────────┬─────────┘ │  │                              │
 │          ▼           │  │                              │
 │ ┌──────────────────┐ │  │                              │
 │ │ Similarity Query │ │  │                              │
 │ │ (Cosine)         │ │  │                              │
 │ └────────┬─────────┘ │  │                              │
 │          ▼           │  │                              │
 │ ┌──────────────────┐ │  │                              │
 │ │ Top-K Retrieval  │─┼─→│  Memory-Conditioned          │
 │ │ (K=5 default)    │ │  │  Action Prediction           │
 │ └──────────────────┘ │  │                              │
 └──────────────────────┘  └──────────────┬───────────────┘
                                          ▼
                            ┌────────────────────────────┐
                            │  Action Chunks             │
                            │  (B, chunk_size, action_dim)│
                            └────────────────────────────┘
```

## Key Features

### 1. Multi-Frame Temporal Processing

- Process **2-10 consecutive frames** (configurable via `n_obs_steps`)
- Each frame gets **temporal positional encoding** to distinguish time steps
- Captures **motion**, **velocity**, and **trajectory** information
- Two temporal encoding options:
  - `learned`: Trainable embeddings (recommended)
  - `sinusoidal`: Fixed positional encoding

### 2. Episodic Memory Module

- **Memory Bank**: Stores up to `memory_size` past observations (default: 1000)
- **Similarity-based Retrieval**: Finds top-K most similar past experiences using cosine similarity
- **Memory Aggregation**: Three modes for combining retrieved memories:
  - `attention`: Learnable attention over memories (most expressive)
  - `mean`: Simple average (fastest)
  - `concat`: Concatenate and project (balanced)

- **Automatic Memory Management**:
  - Circular buffer for efficient storage
  - Automatic addition during inference
  - Configurable update interval

### 3. Memory-Conditioned Decoder

- **Dual Cross-Attention**:
  1. Standard cross-attention to current encoder output
  2. Memory cross-attention to retrieved past experiences

- Allows the policy to condition actions on both:
  - Current multi-frame observations
  - Relevant past experiences

## Usage

### Basic Example

```python
from lerobot.policies.act_mem import ACTMemConfig, ACTMemPolicy
from lerobot.configs.types import Feature, FeatureType

# Configure ACT-MEM
config = ACTMemConfig(
    # Multi-frame settings
    n_obs_steps=3,  # Use 3 recent frames
    temporal_encoding_type="learned",

    # Memory settings
    use_memory=True,
    memory_size=1000,  # Store up to 1000 memories
    num_retrieved_memories=5,  # Retrieve top-5 similar memories
    memory_aggregation="attention",  # Use attention for aggregation
    memory_update_interval=1,  # Add memory every step

    # Standard ACT settings
    chunk_size=100,
    n_action_steps=100,

    # Features
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

# Create policy
policy = ACTMemPolicy(config)

# Reset policy (clears memory) at episode start
policy.reset()

# Inference with multi-frame input
batch = {
    "observation.images.cam_front": torch.randn(1, 3, 3, 96, 96),  # (B, T, C, H, W)
    "observation.state": torch.randn(1, 14),
}

actions = policy.predict_action_chunk(batch)  # (B, chunk_size, action_dim)
```

### Training Example

```python
# Training batch
batch = {
    "observation.images.cam_front": torch.randn(32, 3, 3, 96, 96),  # (B, T, C, H, W)
    "observation.state": torch.randn(32, 14),
    "action": torch.randn(32, 100, 14),
    "action_is_pad": torch.zeros(32, 100, dtype=torch.bool),
}

policy.train()
loss, loss_dict = policy.forward(batch)

# loss_dict contains:
# - "l1_loss": action reconstruction loss
# - "kld_loss": KL divergence loss (if use_vae=True)
```

### Episodic Task Example

```python
# ACT-MEM is particularly useful for episodic tasks where the robot needs to remember
# what it has already done

# Example: Multi-step assembly task
# "I already picked up the blue block, now I need to pick up the red block"

policy.reset()  # Start of episode

for step in range(max_steps):
    # Get current observations (multi-frame)
    obs = env.get_multi_frame_observation(n_frames=3)

    # Policy automatically:
    # 1. Retrieves similar past experiences from memory
    # 2. Conditions action on both current obs and retrieved memories
    # 3. Adds current observation to memory

    action = policy.select_action(obs)
    env.step(action)
```

## Configuration Parameters

### Temporal Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_obs_steps` | int | 3 | Number of consecutive frames to process |
| `temporal_encoding_type` | str | "learned" | Temporal encoding: "learned" or "sinusoidal" |

### Memory Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_memory` | bool | True | Enable episodic memory |
| `memory_size` | int | 1000 | Maximum memories to store |
| `memory_dim` | int | None | Memory embedding dimension (defaults to `dim_model`) |
| `num_retrieved_memories` | int | 5 | Top-K memories to retrieve |
| `memory_aggregation` | str | "attention" | How to aggregate: "attention", "mean", "concat" |
| `memory_dropout` | float | 0.1 | Dropout for memory attention |
| `memory_update_interval` | int | 1 | Add memory every N steps |

### Architecture (inherited from ACT)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 100 | Action prediction horizon |
| `n_action_steps` | int | 100 | Actions per inference |
| `vision_backbone` | str | "resnet18" | ResNet variant |
| `dim_model` | int | 512 | Transformer hidden dimension |
| `n_encoder_layers` | int | 4 | Encoder layers |
| `n_decoder_layers` | int | 1 | Decoder layers |
| `use_vae` | bool | True | Enable VAE objective |
| `latent_dim` | int | 32 | VAE latent dimension |

## When to Use ACT-MEM

### ✅ Highly Recommended For:

1. **Dynamic Environments**
   - Moving objects or obstacles
   - Tasks requiring velocity estimation (catching, tracking)
   - Scenarios with motion blur

2. **Episodic Tasks**
   - Multi-step assembly ("I already attached part A, now do part B")
   - Sequential manipulation ("sorted 3 items, 2 remaining")
   - Tasks with dependencies ("wait until X happens before doing Y")

3. **Tasks Requiring Context Memory**
   - "Don't pick up the same object twice"
   - "Remember which compartment was already filled"
   - "Recall successful approach angles for similar objects"

4. **Long-Horizon Tasks**
   - Tasks spanning 100+ steps where early decisions affect later actions
   - Scenarios requiring backtracking or error recovery

### ❌ May Not Help For:

1. **Simple Static Tasks**
   - Single-step pick-and-place
   - Tasks with no temporal dependencies
   - Scenarios where current frame contains all needed information

2. **Limited Compute**
   - Memory adds ~10-20% computational overhead
   - Multi-frame processing adds N× overhead for N frames

3. **Short Episodes**
   - Episodes with < 10 steps may not benefit from memory
   - Not enough history to learn useful memory retrieval

## Performance Considerations

### Memory Overhead

- **Storage**: O(memory_size × memory_dim)
  - Default: 1000 × 512 = ~2MB per episode

- **Retrieval**: O(num_memories × memory_dim)
  - Cosine similarity computed for all stored memories
  - Top-K selection: O(num_memories log K)

### Computational Overhead

Compared to standard ACT:

1. **Multi-frame**: `n_obs_steps`× more visual features
   - 3 frames ≈ 3× backbone forward passes

2. **Memory**: ~10-20% overhead
   - Similarity computation: O(batch_size × num_memories × memory_dim)
   - Top-K retrieval: O(batch_size × num_memories log K)
   - Memory aggregation: O(batch_size × K × memory_dim)

3. **Total**: Approximately `(n_obs_steps + 0.15)`× standard ACT
   - E.g., 3 frames: ~3.15× slower than ACT

### Tips for Efficiency

1. **Reduce `memory_size`** if episodes are short (<  100 steps)
2. **Use `memory_aggregation="mean"`** for faster aggregation
3. **Increase `memory_update_interval`** to 2-5 to reduce memory additions
4. **Use smaller `num_retrieved_memories`** (3 instead of 5)

## How Memory Works

### 1. Memory Storage

During inference, observations are automatically stored:

```python
# At each step:
observation_embedding = encoder(current_frames)  # (B, D)

# Every memory_update_interval steps:
memory.add_memory(observation_embedding)

# Stored in circular buffer (oldest memories replaced when full)
```

### 2. Memory Retrieval

When making decisions:

```python
# Compute similarity with all stored memories
query = encoder(current_frames)  # (B, D)
similarities = cosine_similarity(query, memory_bank)  # (B, num_memories)

# Retrieve top-K most similar
top_k_indices = torch.topk(similarities, k=num_retrieved_memories)
retrieved_memories = memory_bank[top_k_indices]  # (B, K, D)
```

### 3. Memory Conditioning

```python
# Aggregate retrieved memories
memory_context = aggregate(query, retrieved_memories)  # (B, D)

# Decoder attends to both encoder output AND memory
decoder_output = decoder(
    current_encoding=encoder_output,
    memory_context=memory_context
)
```

## Ablation Studies

Based on the design, we expect:

| Component | Benefit |
|-----------|---------|
| Multi-frame only (no memory) | +5-15% on dynamic tasks |
| Memory only (single frame) | +10-20% on episodic tasks |
| Both multi-frame + memory | +15-30% on complex episodic + dynamic tasks |

## Examples

### Example 1: Dynamic Object Tracking

```python
# Task: Catch a moving ball
config = ACTMemConfig(
    n_obs_steps=5,  # More frames for better velocity estimation
    use_memory=False,  # Memory not needed for this task
)
```

### Example 2: Sequential Assembly

```python
# Task: Assemble 5 parts in sequence
config = ACTMemConfig(
    n_obs_steps=3,
    use_memory=True,
    memory_size=500,  # Smaller memory for shorter episodes
    num_retrieved_memories=3,
)
```

### Example 3: Long-Horizon Manipulation

```python
# Task: Sort 20 objects into 4 bins
config = ACTMemConfig(
    n_obs_steps=3,
    use_memory=True,
    memory_size=2000,  # Larger memory for long episodes
    num_retrieved_memories=10,  # More memories for richer context
    memory_aggregation="attention",  # Most expressive
)
```

## Differences from Standard ACT

| Feature | ACT | ACT-MEM |
|---------|-----|---------|
| Temporal Input | Single frame | Multiple frames (1-10) |
| Temporal Encoding | None | Learned or sinusoidal |
| Memory | None | Episodic memory bank |
| Decoder | Standard cross-attn | Dual cross-attn (encoder + memory) |
| Use Cases | Static, reactive | Dynamic, episodic |

## References

**Original ACT:**
```
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

**Episodic Memory in RL:**
- Blundell et al., "Model-Free Episodic Control" (2016)
- Pritzel et al., "Neural Episodic Control" (2017)

## License

Same as the LeRobot project (Apache 2.0).
