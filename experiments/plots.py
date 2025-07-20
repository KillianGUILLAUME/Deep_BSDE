import matplotlib.pyplot as plt
import torch
import yaml

from bsde.models import HestonSimulator

# Charger la config
with open("experiments/config.yaml") as f:
    config = yaml.safe_load(f)

# Paramètres
T = config["T"]
n_steps = config["n_steps"]
n_paths = config["batch"]
params = config["model"]

# Simulateur Heston
simulator = HestonSimulator(**params, device="cpu")


def plot_heston_paths(n_paths=500, n_steps=252, T=1.0):
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


plot_heston_paths(n_paths=50000, n_steps=252, T=T)
