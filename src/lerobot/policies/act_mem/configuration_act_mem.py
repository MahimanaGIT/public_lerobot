#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_mem")
@dataclass
class ACTMemConfig(PreTrainedConfig):
    """Configuration class for the ACT-MEM (ACT with Memory) policy.

    ACT-MEM extends the Action Chunking Transformer with two key improvements:
    1. Multi-frame temporal processing for velocity/motion estimation
    2. Episodic memory for remembering key decisions and past experiences

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.
        - Images should have shape (n_obs_steps, C, H, W) when n_obs_steps > 1

    Args:
        # Temporal processing (for velocity/motion estimation)
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
            Default is 3 for multi-frame processing.
        temporal_encoding_type: Type of temporal positional encoding to use.
            Options: "learned" (trainable embeddings) or "sinusoidal" (fixed encoding).

        # Memory settings (for episodic memory)
        use_memory: Whether to enable the episodic memory module.
        memory_size: Maximum number of memories to store in the memory bank.
        memory_dim: Dimension of memory embeddings (default same as dim_model).
        num_retrieved_memories: Number of top-K memories to retrieve and attend to.
        memory_aggregation: How to aggregate retrieved memories. Options:
            - "attention": Learnable attention over retrieved memories
            - "mean": Simple average of retrieved memories
            - "concat": Concatenate and project retrieved memories
        memory_dropout: Dropout rate for memory attention.
        memory_update_interval: How often to add new memories (in steps). 1 = every step.

        # Standard ACT parameters
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        input_shapes: Dictionary defining shapes of input data.
        output_shapes: Dictionary defining shapes of output data.
        vision_backbone: Name of the torchvision resnet backbone.
        pretrained_backbone_weights: Pretrained weights from torchvision.
        replace_final_stride_with_dilation: Whether to replace ResNet's final stride with dilation.
        pre_norm: Whether to use "pre-norm" in transformer blocks.
        dim_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        dim_feedforward: Feed-forward expansion dimension.
        feedforward_activation: Activation function for feed-forward layers.
        n_encoder_layers: Number of transformer encoder layers.
        n_decoder_layers: Number of transformer decoder layers.
        use_vae: Whether to use variational objective during training.
        latent_dim: VAE latent dimension.
        n_vae_encoder_layers: Number of VAE encoder layers.
        temporal_ensemble_coeff: Coefficient for temporal ensembling (optional).
        dropout: Dropout rate for transformer layers.
        kl_weight: Weight for KL-divergence loss component.
    """

    # Input / output structure.
    n_obs_steps: int = 3  # Multi-frame for velocity estimation
    chunk_size: int = 100
    n_action_steps: int = 100

    # Temporal encoding for multi-frame
    temporal_encoding_type: str = "learned"  # or "sinusoidal"

    # Memory settings
    use_memory: bool = True
    memory_size: int = 1000  # Number of memories to store
    memory_dim: int | None = None  # If None, use dim_model
    num_retrieved_memories: int = 5  # Top-K memories to retrieve
    memory_aggregation: str = "attention"  # "attention", "mean", or "concat"
    memory_dropout: float = 0.1
    memory_update_interval: int = 1  # Add memory every N steps

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        # Set memory_dim to dim_model if not specified
        if self.memory_dim is None:
            self.memory_dim = self.dim_model

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps < 1:
            raise ValueError(
                f"`n_obs_steps` must be at least 1. Got `n_obs_steps={self.n_obs_steps}`"
            )
        if self.temporal_encoding_type not in ["learned", "sinusoidal"]:
            raise ValueError(
                f"`temporal_encoding_type` must be either 'learned' or 'sinusoidal'. "
                f"Got {self.temporal_encoding_type}."
            )
        if self.memory_aggregation not in ["attention", "mean", "concat"]:
            raise ValueError(
                f"`memory_aggregation` must be one of 'attention', 'mean', or 'concat'. "
                f"Got {self.memory_aggregation}."
            )
        if self.use_memory and self.memory_size < 1:
            raise ValueError(
                f"`memory_size` must be at least 1 when using memory. Got {self.memory_size}."
            )
        if self.use_memory and self.num_retrieved_memories < 1:
            raise ValueError(
                f"`num_retrieved_memories` must be at least 1 when using memory. "
                f"Got {self.num_retrieved_memories}."
            )
        if self.num_retrieved_memories > self.memory_size:
            raise ValueError(
                f"`num_retrieved_memories` ({self.num_retrieved_memories}) cannot be greater than "
                f"`memory_size` ({self.memory_size})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
