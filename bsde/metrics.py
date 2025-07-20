import torch
import yaml

with open("experiments/config.yaml") as f:
    config = yaml.safe_load(f)


@torch.no_grad()
def delta_hedge(
    simulator,
    net,
    payoff_fn,
    n_paths=config["batch"],
    n_steps=config["n_steps"],
    T=1.0,
    r=0.0,
    device="cpu",
):
    """return p&l distribution

    args:
        simulator: HestonSimulator
        net: FeedForwardNN
        payoff_fn: callable
        n_paths (int): Number of paths to simulate.
        n_steps (int): Number of time steps in each path.
        T (float): Time horizon for the simulation.
        r (float): Risk-free rate, used for discounting in payoff calculation.
    """

    S, v = simulator.sample_paths(n_paths, n_steps, T)
    dt = T / n_steps

    t0 = torch.zeros(n_paths, device=device)

    y, z = net(t0, S[0], v[0])

    print(f"Y0 net  :, {y[0].item()}, should be : MC Price = 0.13")
    print(f"Delta net :, {z[0,0].item()}, should be : << 1")
    print()

    pos = z[:, 0]

    cash = y - pos * S[0]

    for k in range(n_steps):
        if r:
            cash *= torch.exp(torch.tensor(-r * T, device=device))

        t_k = torch.full((n_paths,), k * dt, device=device)
        y, z = net(t_k, S[k], v[k])
        new_delta = z[:, 0]

        cash -= (new_delta - pos) * S[k]
        pos = new_delta

    if r:
        cash *= torch.exp(torch.tensor(-r * T, device=device))

    pnl = cash + pos * S[-1] - payoff_fn(S)

    payoff = payoff_fn(S)
    print("payoff  mean :", payoff.mean().item())
    print("payoff  max  :", payoff.max().item(), "   payoff min :", payoff.min().item())

    print("cash0  mean  :", cash.mean().item())
    print("pos0   mean  :", pos.mean().item())
    return pnl.cpu()


@torch.no_grad()
def dv_hedge(
    simulator,
    net,
    payoff_fn,
    n_paths=config["batch"],
    n_steps=config["n_steps"],
    T=config["T"],
    r=0.0,
    device="cpu",
):
    """return p&l distribution
    args:
        simulator: HestonSimulator
        net: FeedForwardNN
        payoff_fn: callable
        n_paths (int): Number of paths to simulate.
        n_steps (int): Number of time steps in each path.
        T (float): Time horizon for the simulation.
        r (float): Risk-free rate, used for discounting in payoff calculation.
    """
    S, v = simulator.sample_paths(n_paths, n_steps, T)
    dt = T / n_steps

    t0 = torch.zeros(n_paths, device=device)
    y, z = net(t0, S[0], v[0])
    pos_s = z[:, 0]
    pos_v = z[:, 1]

    cash = y - pos_s * S[0] - pos_v * v[0]

    for k in range(1, n_steps):
        if r:
            cash *= torch.exp(torch.tensor(-r * dt, device=device))

        t_k = torch.full((n_paths,), k * dt, device=device)
        y, z = net(t_k, S[k], v[k])

        new_pos_s = z[:, 0]
        new_pos_v = z[:, 1]

        cash -= (new_pos_s - pos_s) * S[k] + (new_pos_v - pos_v) * v[k]
        pos_s = new_pos_s
        pos_v = new_pos_v

        if r:
            cash *= torch.exp(torch.tensor(-r * dt, device=device))

        pnl = cash + pos_s * S[-1] + pos_v * v[-1] - payoff_fn(S)

        return pnl.cpu()
