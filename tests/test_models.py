import torch

from bsde.models import HestonSimulator


def test_shapes():
    simulator = HestonSimulator(
        mu=0.05,
        kappa=1.0,
        theta=0.04,
        sigma=0.2,
        rho=-0.5,
        s0=100.0,
        v0=0.04,
        device="cpu",
    )

    n_paths = 100
    n_steps = 1000
    T = 1.0

    S, v = simulator.sample_paths(n_paths, n_steps, T)

    assert S.shape == (
        n_steps + 1,
        n_paths,
    ), f"Expected S shape {(n_steps + 1, n_paths)}, got {S.shape}"
    assert v.shape == (
        n_steps + 1,
        n_paths,
    ), f"Expected v shape {(n_steps + 1, n_paths)}, got {v.shape}"
    assert torch.all(v >= 0), "Variance v must be non-negative"
