#!/usr/bin/env python
"""Simple test script for ACT Multi-Frame policy."""

import torch
from lerobot.policies.act_multiframe import ACTMultiFrameConfig, ACTMultiFramePolicy
from lerobot.configs.types import Feature, FeatureType

def test_act_multiframe_basic():
    """Test basic instantiation and forward pass of ACT Multi-Frame policy."""

    # Create configuration with multi-frame support
    config = ACTMultiFrameConfig(
        n_obs_steps=3,  # Use 3 frames
        chunk_size=10,
        n_action_steps=10,
        input_features={
            "observation.images.cam_front": Feature(
                shape=[3, 3, 96, 96],  # (n_obs_steps, C, H, W)
                dtype="float32",
                name="observation.images.cam_front",
                feature_type=FeatureType.VISUAL,
            ),
            "observation.state": Feature(
                shape=[14],  # Robot state
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
        device="cpu",
    )

    # Instantiate policy
    print("Creating ACT Multi-Frame policy...")
    policy = ACTMultiFramePolicy(config)
    print(f"Policy created successfully with {sum(p.numel() for p in policy.parameters())} parameters")

    # Create dummy input batch
    batch_size = 2
    batch = {
        "observation.images.cam_front": torch.randn(batch_size, 3, 3, 96, 96),  # (B, T, C, H, W)
        "observation.state": torch.randn(batch_size, 14),  # (B, state_dim)
        "action": torch.randn(batch_size, config.chunk_size, 14),  # (B, chunk_size, action_dim)
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }

    # Test forward pass (training mode)
    print("\nTesting forward pass (training mode)...")
    policy.train()
    loss, loss_dict = policy.forward(batch)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    # Test inference
    print("\nTesting inference mode...")
    policy.eval()
    with torch.no_grad():
        # Remove action from batch for inference
        inference_batch = {
            "observation.images.cam_front": batch["observation.images.cam_front"],
            "observation.state": batch["observation.state"],
        }
        actions = policy.predict_action_chunk(inference_batch)
        print(f"Predicted actions shape: {actions.shape}")
        assert actions.shape == (batch_size, config.chunk_size, 14), \
            f"Expected shape ({batch_size}, {config.chunk_size}, 14), got {actions.shape}"

    print("\n✅ All tests passed!")


def test_act_multiframe_single_frame_compat():
    """Test that the policy also works with single frames (n_obs_steps=1)."""

    config = ACTMultiFrameConfig(
        n_obs_steps=1,  # Single frame
        chunk_size=10,
        n_action_steps=10,
        input_features={
            "observation.images.cam_front": Feature(
                shape=[3, 96, 96],  # (C, H, W) - single frame
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
        device="cpu",
    )

    print("\nTesting single-frame compatibility...")
    policy = ACTMultiFramePolicy(config)

    batch_size = 2
    batch = {
        "observation.images.cam_front": torch.randn(batch_size, 3, 96, 96),  # (B, C, H, W)
        "observation.state": torch.randn(batch_size, 14),
        "action": torch.randn(batch_size, config.chunk_size, 14),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }

    policy.train()
    loss, loss_dict = policy.forward(batch)
    print(f"Single-frame loss: {loss.item():.4f}")

    print("✅ Single-frame compatibility test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ACT Multi-Frame Policy")
    print("=" * 60)

    test_act_multiframe_basic()
    test_act_multiframe_single_frame_compat()

    print("\n" + "=" * 60)
    print("All tests completed successfully! 🎉")
    print("=" * 60)
