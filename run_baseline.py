import jax
import matplotlib.pyplot as plt
from simulator import run_simulation
from plotter import plot_results


if __name__ == "__main__":
    history, leader_history = run_simulation(feedforward=False)
    plot_results(history, leader_history, "Baseline Rendezvous (Consensus Only, No Feedforward)")
