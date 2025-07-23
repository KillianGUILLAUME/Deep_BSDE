This repo aims to make an implementation of the deep-bsde method, coming from the paper by Han & Long (2018), arXiv link at the bottom of the page.


First, we use this method to estimate, via a neural network, the path of a 1-D Heston model via a lookback option.
The neural network is implemented in PyTorch using TensorBoard logging.

### Quick start ###

script file run the neural network,

from bsde, we get :
-metrics : contains delta_hedge and delta_hedge_vega functions, in order to check if the path we extract from the nn is well approximated.
-models : contains multi dimensional Black-Scholes and Heston 1-D models.
-nets : creat the neural network. This is a feedforward model, with the depth and the number of layer as parameters.
-payoff : lookback and baskets calls functions.

from experiments, we have :
-plot_path_ : plot different path for Heston Model
-grid_search : tests different combinations of learning rates and batch_size (number of days)

-gs_notebook_plot : file to analyze the results from "grid_search" in experiments.

-delta_hedge : to ensure that the Y_0 value learnt by the network is supposed to be the option's risk-neutral value.

TODO :
Black-Scholes d-dimensional, and run the neural network, compare it with the monte-carlo methods.
Do a grid search with several dimensions, plot the convergence rates in terms of times.

Optimize parameters with Ray Tune or Optuna.

Implément a C++ section to speed-up the process, or run it on a CUDA / GPU.

If you use this repo, please cite:
@misc{guillaume2025deepbsde,
  title   = {Deep BSDE Option Pricing},
  author  = {Guillaume, Killian},
  year    = {2025},
  url     = {https://github.com/KillianGUILLAUME/Deep_BSDE}
}
