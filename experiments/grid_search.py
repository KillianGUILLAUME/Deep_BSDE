import itertools
import time
from pathlib import Path

import pandas as pd
import yaml

from benchmark import monte_carlo_sim
from bsde.payoff import lookback_max
from bsde.solver import BSDESolver

cfg = yaml.safe_load(Path("experiments/config.yaml").read_text())


def payoff_func(S):
    """Look-back payoff used in the grid-search."""
    return lookback_max(S, strike=cfg["payoff"]["K"])


grid = list(itertools.product([10, 30, 60, 90, 150, 252, 504], [1e-3, 1e-4, 1e-5]))
results = []

for n_steps, lr in grid:
    t0 = time.perf_counter()
    solver = BSDESolver(
        model_params=cfg["model"],
        payoff_fn=payoff_func,
        T=cfg["T"],
        n_steps=n_steps,
        net_cfg=cfg["net"],
        lr=lr,
        device="cpu",
    )
    solver.fit(n_epochs=cfg["epochs"], batch_size=cfg["batch"])
    price = solver.predict()
    final_loss = solver.fwd_one_batch(n_paths=cfg["batch"])[0].item()
    runtime = time.perf_counter() - t0

    mc_price, mc_ci = monte_carlo_sim(
        model=solver.simulator,
        payoff_fn=payoff_func,
        n_paths=cfg["batch"],
        n_steps=n_steps,
        T=cfg["T"],
    )

    err = (price - mc_price) ** 2

    results.append(
        dict(
            n_steps=n_steps,
            lr=lr,
            price=price,
            mc_price=mc_price,
            final_loss=final_loss,
            err=err,
            mc_ci=mc_ci,
            times=runtime,
        )
    )

    print(
        f"n_steps={n_steps}, lr={lr}, price={price:.4f}, mc_price={mc_price:.4f}, "
        f"final_loss={final_loss:.4e}, err={err:.3e}, mc_ci={mc_ci:.4f}, time={runtime:.2f}s"
    )

df = pd.DataFrame(results)
df.to_csv("results/grid_search_results.csv", index=False)
print("\nResults :\n", df)
