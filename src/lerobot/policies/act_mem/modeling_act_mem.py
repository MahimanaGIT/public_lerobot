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
"""ACT-MEM: Action Chunking Transformer with Multi-frame and Episodic Memory

Combines two key improvements over standard ACT:
1. Multi-frame temporal processing for velocity/motion estimation
2. Episodic memory for remembering key decisions and past experiences
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_mem.configuration_act_mem import ACTMemConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTMemPolicy(PreTrainedPolicy):
    """
    ACT-MEM: Action Chunking Transformer with Multi-frame and Episodic Memory

    Extends ACT with:
    1. Multi-frame temporal processing for velocity/motion estimation
    2. Episodic memory to remember key past experiences and decisions
    """

    config_class = ACTMemConfig
    name = "act_mem"

    def __init__(
        self,
        config: ACTMemConfig,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTMem(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        # Reset the action queue
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

        # Reset the memory module
        if self.config.use_memory:
            self.model.memory.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705."""
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class EpisodicMemory(nn.Module):
    """Episodic memory module for storing and retrieving past experiences.

    Stores embeddings of past observations and actions, and retrieves the most
    relevant memories based on similarity to the current observation.
    """

    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim
        self.num_retrieved = config.num_retrieved_memories

        # Memory bank buffers (not trainable parameters)
        self.register_buffer("memory_keys", torch.zeros(config.memory_size, config.memory_dim))
        self.register_buffer("memory_values", torch.zeros(config.memory_size, config.memory_dim))
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("memory_count", torch.zeros(1, dtype=torch.long))

        # Projection layers for memory
        self.query_proj = nn.Linear(config.dim_model, config.memory_dim)
        self.key_proj = nn.Linear(config.dim_model, config.memory_dim)
        self.value_proj = nn.Linear(config.dim_model, config.memory_dim)

        self.step_counter = 0

    def reset(self):
        """Reset memory bank (called at episode start)."""
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_ptr.zero_()
        self.memory_count.zero_()
        self.step_counter = 0

    def add_memory(self, observation_embedding: Tensor):
        """Add a new memory to the bank.

        Args:
            observation_embedding: (B, D) or (D,) tensor of observation embeddings
        """
        if observation_embedding.dim() == 2:
            # Batch dimension present, take first element
            observation_embedding = observation_embedding[0]

        # Project to memory space
        key = self.key_proj(observation_embedding)
        value = self.value_proj(observation_embedding)

        # Add to circular buffer
        ptr = int(self.memory_ptr.item())
        self.memory_keys[ptr] = key.detach()
        self.memory_values[ptr] = value.detach()

        # Update pointer (circular)
        self.memory_ptr[0] = (ptr + 1) % self.memory_size
        self.memory_count[0] = min(self.memory_count[0] + 1, self.memory_size)

    def retrieve_memories(self, query_embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Retrieve top-K most similar memories.

        Args:
            query_embedding: (B, D) current observation embedding

        Returns:
            retrieved_keys: (B, K, D) retrieved memory keys
            retrieved_values: (B, K, D) retrieved memory values
        """
        batch_size = query_embedding.shape[0]

        # Project query
        query = self.query_proj(query_embedding)  # (B, memory_dim)

        # Get valid memories
        num_memories = int(self.memory_count.item())
        if num_memories == 0:
            # No memories yet, return zeros
            device = query.device
            return (
                torch.zeros(batch_size, self.num_retrieved, self.memory_dim, device=device),
                torch.zeros(batch_size, self.num_retrieved, self.memory_dim, device=device),
            )

        # Compute similarities (cosine similarity)
        valid_keys = self.memory_keys[:num_memories]  # (num_memories, D)
        valid_values = self.memory_values[:num_memories]  # (num_memories, D)

        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)  # (B, D)
        keys_norm = F.normalize(valid_keys, dim=-1)  # (num_memories, D)

        # Compute similarity
        similarity = torch.matmul(query_norm, keys_norm.t())  # (B, num_memories)

        # Get top-K
        k = min(self.num_retrieved, num_memories)
        top_k_values, top_k_indices = torch.topk(similarity, k, dim=-1)  # (B, K)

        # Retrieve memories
        retrieved_keys = valid_keys[top_k_indices]  # (B, K, D)
        retrieved_values = valid_values[top_k_indices]  # (B, K, D)

        # Pad if necessary
        if k < self.num_retrieved:
            padding = self.num_retrieved - k
            pad_keys = torch.zeros(
                batch_size, padding, self.memory_dim,
                device=query.device, dtype=query.dtype
            )
            pad_values = torch.zeros(
                batch_size, padding, self.memory_dim,
                device=query.device, dtype=query.dtype
            )
            retrieved_keys = torch.cat([retrieved_keys, pad_keys], dim=1)
            retrieved_values = torch.cat([retrieved_values, pad_values], dim=1)

        return retrieved_keys, retrieved_values

    def forward(self, observation_embedding: Tensor, add_to_memory: bool = False):
        """
        Args:
            observation_embedding: (B, D) current observation embedding
            add_to_memory: Whether to add this observation to memory

        Returns:
            retrieved_values: (B, K, D) retrieved memory values for attention
        """
        # Retrieve memories
        retrieved_keys, retrieved_values = self.retrieve_memories(observation_embedding)

        # Optionally add to memory (during rollout/evaluation)
        if add_to_memory:
            self.step_counter += 1
            if self.step_counter % self.config.memory_update_interval == 0:
                self.add_memory(observation_embedding)

        return retrieved_values


class MemoryAggregation(nn.Module):
    """Aggregates retrieved memories for the decoder."""

    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.config = config
        self.aggregation_type = config.memory_aggregation

        if self.aggregation_type == "attention":
            # Learnable attention over memories
            self.attention = nn.MultiheadAttention(
                config.memory_dim,
                config.n_heads,
                dropout=config.memory_dropout,
                batch_first=True
            )
        elif self.aggregation_type == "concat":
            # Concatenate and project
            self.projection = nn.Linear(
                config.memory_dim * config.num_retrieved_memories,
                config.dim_model
            )

    def forward(self, query: Tensor, memories: Tensor) -> Tensor:
        """
        Args:
            query: (B, dim_model) current embedding
            memories: (B, K, memory_dim) retrieved memories

        Returns:
            aggregated: (B, dim_model) aggregated memory representation
        """
        if self.aggregation_type == "attention":
            # Use query to attend over memories
            # Expand query for attention
            query_expanded = query.unsqueeze(1)  # (B, 1, D)
            attended, _ = self.attention(
                query_expanded, memories, memories
            )  # (B, 1, D)
            return attended.squeeze(1)  # (B, D)

        elif self.aggregation_type == "mean":
            # Simple average
            return memories.mean(dim=1)  # (B, memory_dim)

        elif self.aggregation_type == "concat":
            # Concatenate and project
            batch_size = memories.shape[0]
            concatenated = memories.view(batch_size, -1)  # (B, K * memory_dim)
            return self.projection(concatenated)  # (B, dim_model)

        else:
            raise ValueError(f"Unknown memory aggregation type: {self.aggregation_type}")


class ACTMem(nn.Module):
    """ACT with Multi-frame and Episodic Memory.

    Extends ACT architecture with:
    1. Multi-frame temporal processing
    2. Episodic memory module
    3. Memory-conditioned decoder
    """

    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.config = config

        # VAE Encoder (same as ACT)
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Vision backbone
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer encoder
        self.encoder = ACTEncoder(config)

        # Memory module
        if config.use_memory:
            self.memory = EpisodicMemory(config)
            self.memory_aggregation = MemoryAggregation(config)
            # Memory decoder with cross-attention to both encoder and memory
            self.decoder = ACTMemDecoder(config)
        else:
            self.decoder = ACTDecoder(config)

        # Input projections
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # Positional embeddings
        n_1d_tokens = 1  # latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        if self.config.image_features:
            # Spatial positional encoding
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
            # Temporal positional encoding
            if config.temporal_encoding_type == "learned":
                self.temporal_pos_embed = nn.Embedding(config.n_obs_steps, config.dim_model)
            elif config.temporal_encoding_type == "sinusoidal":
                self.register_buffer(
                    "temporal_pos_embed",
                    create_sinusoidal_pos_embedding(config.n_obs_steps, config.dim_model)
                )

        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Action head
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through ACT-MEM.

        Returns:
            (B, chunk_size, action_dim) action predictions
            Tuple of (mu, log_sigma_x2) for VAE or (None, None)
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using VAE in training mode."

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # VAE encoder (same as ACT)
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state = batch[OBS_STATE]
                if robot_state.dim() == 3:
                    robot_state = robot_state[:, -1, :]
                robot_state_embed = self.vae_encoder_robot_state_input_proj(robot_state).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            vae_encoder_input = [cls_embed, robot_state_embed, action_embed] if self.config.robot_state_feature else [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device if OBS_STATE in batch else batch[ACTION].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            device = batch[OBS_IMAGES][0].device if OBS_IMAGES in batch else batch[OBS_ENV_STATE].device
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

        # Prepare encoder inputs (multi-frame)
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            robot_state = batch[OBS_STATE]
            if robot_state.dim() == 3:
                robot_state = robot_state[:, -1, :]
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(robot_state))

        if self.config.env_state_feature:
            env_state = batch[OBS_ENV_STATE]
            if env_state.dim() == 3:
                env_state = env_state[:, -1, :]
            encoder_in_tokens.append(self.encoder_env_state_input_proj(env_state))

        # Process multi-frame images
        if self.config.image_features:
            for cam_idx, img_seq in enumerate(batch[OBS_IMAGES]):
                if img_seq.dim() == 4:
                    img_seq = img_seq.unsqueeze(1)

                batch_size, n_steps, C, H, W = img_seq.shape

                for t in range(n_steps):
                    img = img_seq[:, t]
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)

                    # Add temporal encoding
                    if self.config.temporal_encoding_type == "learned":
                        temporal_embed = self.temporal_pos_embed.weight[t].view(1, -1, 1, 1)
                    else:
                        temporal_embed = self.temporal_pos_embed[t].view(1, -1, 1, 1).to(cam_features.device)
                    cam_features = cam_features + temporal_embed

                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack tokens
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Transformer encoder
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # Retrieve memories if using memory
        memory_context = None
        if self.config.use_memory:
            # Use first encoder output token (latent) as observation embedding for memory
            obs_embedding = encoder_out[0]  # (B, D)

            # Retrieve memories
            retrieved_memories = self.memory(
                obs_embedding,
                add_to_memory=(not self.training)  # Add to memory during inference
            )

            # Aggregate memories
            memory_context = self.memory_aggregation(obs_embedding, retrieved_memories)

        # Decoder
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )

        if self.config.use_memory:
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                memory_context=memory_context,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )
        else:
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


# Helper classes (ACTEncoder, ACTDecoder, etc.) - reusing from ACT implementation


class ACTEncoder(nn.Module):
    """Multi-layer transformer encoder."""

    def __init__(self, config: ACTMemConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    """Standard ACT decoder (without memory)."""

    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTMemDecoder(nn.Module):
    """Memory-augmented ACT decoder with cross-attention to both encoder and memory."""

    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTMemDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        memory_context: Tensor | None = None,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out,
                memory_context=memory_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class ACTMemDecoderLayer(nn.Module):
    """Decoder layer with cross-attention to both encoder output and memory."""

    def __init__(self, config: ACTMemConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.encoder_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.memory_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.norm4 = nn.LayerNorm(config.dim_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.dropout4 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        memory_context: Tensor | None = None,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # Self-attention
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        # Cross-attention to encoder
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.encoder_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        # Cross-attention to memory (if available)
        if memory_context is not None:
            if self.pre_norm:
                skip = x
                x = self.norm3(x)
            else:
                x = self.norm2(x)
                skip = x

            # Expand memory_context for all decoder positions
            # memory_context: (B, D)
            # x: (S, B, D)
            seq_len, batch_size, dim = x.shape
            memory_expanded = memory_context.unsqueeze(0).expand(1, batch_size, dim)  # (1, B, D)

            x = self.memory_attn(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=memory_expanded,
                value=memory_expanded,
            )[0]
            x = skip + self.dropout3(x)

        # Feed-forward
        if self.pre_norm:
            skip = x
            x = self.norm4(x)
        else:
            x = self.norm3(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout4(x)
        if not self.pre_norm:
            x = self.norm4(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings for spatial feature maps."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
