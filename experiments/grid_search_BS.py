import itertools
import time
from pathlib import Path

import yaml

from benchmark import monte_carlo_sim
from bsde.payoff import basket_call
from bsde.solver import DeepBSDESolverBS

cfg = yaml.safe_load(Path("experiments/config.yaml").read_text())


def payoff_func(S):
    """Basket-call payoff used in the grid-search."""
    return basket_call(S, strike=cfg["payoff"]["K"])


grid = list(itertools.product([10, 25, 50, 100], [1e-4, 1e-5]))  # dim, learning rate
results = []

for dim, lr in grid:
    t0 = time.perf_counter()
    solver = DeepBSDESolverBS(
        model_params=cfg["model"],
        payoff_fn=payoff_func,
        T=cfg["T"],
        n_steps=cfg["n_steps"],
        net_cfg=cfg["net"],
        lr=lr,
        device="cpu",
        dimension=dim,
    )
    solver.fit(n_epochs=cfg["epochs"], batch_size=cfg["batch"])
    price = solver.predict()
    final_loss = solver.fwd_one_batch(n_paths=cfg["batch"])[0].item()
    runtime = time.perf_counter() - t0

    mc_price, mc_ci = monte_carlo_sim(
        model=solver.simulator,
        payoff_fn=payoff_func,
        n_paths=cfg["batch"],
        n_steps=dim,
        T=cfg["T"],
    )

    err = (price - mc_price) ** 2

    results.append(
        dict(
            dim=dim,
            lr=lr,
            price=price,
            mc_price=mc_price,
            final_loss=final_loss,
            err=err,
            mc_ci=mc_ci,
            times=runtime,
        )
    )
