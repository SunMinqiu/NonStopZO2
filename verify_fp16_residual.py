"""
Verify whether fp16 perturbation residuals exist and their magnitude.
This script simulates what happens in zo_forward's perturbation sequence
and compares the result with and without perturbation simulation.
"""
import torch
import numpy as np
import json
import os

def test_single_param_residual():
    """Test perturbation residual on a single fp16 parameter."""
    print("=" * 60)
    print("Test 1: Single parameter fp16 perturbation residual")
    print("=" * 60)

    eps = 0.001
    seed = 42

    # Test various parameter values
    test_values = [0.5, 1.0, 1.5, 2.0, 0.01, 0.001, 10.0]

    for val in test_values:
        param = torch.tensor([val], dtype=torch.float16)
        original = param.clone()

        # Simulate perturbation sequence [+1, -2, +1]
        for scaling in [1, -2, 1]:
            torch.manual_seed(seed)
            z = torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype, device=param.device)
            param.data.add_(scaling * z * eps)

        residual = (param - original).item()
        print(f"  param={val:.4f}, z={z.item():.4f}, residual={residual:.10f}")


def test_model_scale_residual():
    """Simulate 100 steps of perturbation on a realistic-scale parameter tensor."""
    print("\n" + "=" * 60)
    print("Test 2: 100 steps of perturbation on a 1000-dim fp16 tensor")
    print("=" * 60)

    eps = 0.001
    lr = 1e-7

    # Simulate a weight matrix chunk
    torch.manual_seed(0)
    param_original = torch.randn(1000, dtype=torch.float16)

    # Path A: Training (perturbation + update)
    param_train = param_original.clone()
    # Path B: Replay (update only)
    param_replay = param_original.clone()

    grads = []
    seeds = []

    for step in range(100):
        seed = np.random.randint(1, 2**31)
        seeds.append(seed)
        grad = float(np.random.uniform(-100, 100))
        grads.append(grad)

        # Path A: Full training sequence (perturb + update)
        for scaling in [1, -2, 1]:
            torch.manual_seed(seed)
            z = torch.normal(mean=0, std=1, size=param_train.size(), dtype=param_train.dtype, device=param_train.device)
            param_train.data.add_(scaling * z * eps)

        torch.manual_seed(seed)
        z = torch.normal(mean=0, std=1, size=param_train.size(), dtype=param_train.dtype, device=param_train.device)
        param_train.data.sub_(lr * grad * z)

        # Path B: Replay (update only)
        torch.manual_seed(seed)
        z = torch.normal(mean=0, std=1, size=param_replay.size(), dtype=param_replay.dtype, device=param_replay.device)
        param_replay.data.sub_(lr * grad * z)

    diff = (param_train - param_replay).float()
    update_magnitude = (param_replay - param_original).float()

    print(f"  After 100 steps:")
    print(f"    Accumulated update (replay-original) L2: {update_magnitude.norm():.6e}")
    print(f"    Train vs Replay diff L2:                 {diff.norm():.6e}")
    print(f"    Train vs Replay diff max:                {diff.abs().max():.6e}")
    print(f"    Train vs Replay diff mean:               {diff.abs().mean():.6e}")
    print(f"    Ratio (diff / update):                   {diff.norm() / (update_magnitude.norm() + 1e-20):.2f}x")
    print(f"    Non-zero residual elements:              {(diff != 0).sum().item()} / {diff.numel()}")

    # Path C: Replay WITH perturbation simulation (the fix)
    param_fixed = param_original.clone()
    for step in range(100):
        seed = seeds[step]
        grad = grads[step]

        # Simulate perturbation
        for scaling in [1, -2, 1]:
            torch.manual_seed(seed)
            z = torch.normal(mean=0, std=1, size=param_fixed.size(), dtype=param_fixed.dtype, device=param_fixed.device)
            param_fixed.data.add_(scaling * z * eps)

        # Apply update
        torch.manual_seed(seed)
        z = torch.normal(mean=0, std=1, size=param_fixed.size(), dtype=param_fixed.dtype, device=param_fixed.device)
        param_fixed.data.sub_(lr * grad * z)

    diff_fixed = (param_train - param_fixed).float()
    print(f"\n  After fix (replay WITH perturbation simulation):")
    print(f"    Train vs FixedReplay diff L2:            {diff_fixed.norm():.6e}")
    print(f"    Train vs FixedReplay diff max:           {diff_fixed.abs().max():.6e}")
    print(f"    Non-zero residual elements:              {(diff_fixed != 0).sum().item()} / {diff_fixed.numel()}")


def test_with_real_checkpoint():
    """If a real checkpoint exists, test with actual update records."""
    print("\n" + "=" * 60)
    print("Test 3: Real checkpoint comparison (if available)")
    print("=" * 60)

    # Try to find a real checkpoint
    checkpoint_dirs = [
        "/home/users/u0001609/zo_output",
        "/home/users/u0001609/ZO_logs",
    ]

    for d in checkpoint_dirs:
        if os.path.exists(d):
            # Look for optimizer.pt files
            for root, dirs, files in os.walk(d):
                if "optimizer.pt" in files:
                    opt_path = os.path.join(root, "optimizer.pt")
                    try:
                        opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                        if isinstance(opt_state, dict) and 'zo_update_history' in opt_state:
                            updates = opt_state['zo_update_history']
                            print(f"  Found checkpoint: {opt_path}")
                            print(f"  Updates: {len(updates)}")
                            if updates:
                                first = updates[0]
                                print(f"  First update keys: {list(first.keys())}")
                                has_eps = 'zo_eps' in first
                                print(f"  Has zo_eps in records: {has_eps}")
                            return
                    except:
                        pass

    print("  No real checkpoint found, skipping.")


if __name__ == "__main__":
    test_single_param_residual()
    test_model_scale_residual()
    test_with_real_checkpoint()
