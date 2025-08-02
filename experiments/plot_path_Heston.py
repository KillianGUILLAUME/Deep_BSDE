import matplotlib.pyplot as plt
import torch
import yaml

from bsde.models import HestonSimulator

with open("experiments/config.yaml") as f:
    config = yaml.safe_load(f)

# parameters
T = config["T"]
n_steps = config["n_steps"]
n_paths = config["batch"]
params = config["model"]


simulator = HestonSimulator(**params, device="cpu")


def plot_heston_paths(n_paths=50, n_steps=252, T=1.0):
    S, v = simulator.sample_paths(n_paths, n_steps, T)
    t_grid = torch.linspace(0, T, n_steps + 1)

    plt.figure(figsize=(10, 5))
    for i in range(min(n_paths, 50)):
        plt.plot(t_grid, S[:, i].cpu(), lw=1)
    plt.title(f"Heston model - {n_paths} paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Asset Price S(t)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(n_paths):
        plt.plot(t_grid, v[:, i], lw=1)
    plt.title(f"Heston variance – {n_paths} paths (v0 = {params['v0']})")
    plt.xlabel("Time (years)")
    plt.ylabel("Instant. variance v(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_heston_paths(n_paths=n_paths, n_steps=n_steps, T=T)
