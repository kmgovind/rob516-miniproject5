"""strat_comparison.py

Run both the feedforward algorithm and the baseline, compute comparison
metrics, and optionally plot mean-tracking-error over time.

Usage:
    python code/strat_comparison.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from simulator import run_simulation
from plotter import plot_results, comparison_plot, video_plot


def compute_metrics(history, leader_history, dt, threshold=0.5):
    """Compute simple tracking metrics comparing agents to the leader.

    history: array shape (T+1, N, dim)
    leader_history: array shape (T, dim)
    Returns: (metrics_dict, mean_error_time (np.array), errors (np.array T x N))
    """
    history = jnp.asarray(history)
    leader_history = jnp.asarray(leader_history)

    # Align: compare history[:-1] (t=0..T-1) with leader_history (t=0..T-1)
    positions = history[:-1]  # shape (T, N, dim)
    leader = leader_history    # shape (T, dim)

    diffs = positions - leader[:, None, :]
    errors = jnp.linalg.norm(diffs, axis=2)  # shape (T, N)
    mean_error_time = jnp.mean(errors, axis=1)  # shape (T,)

    final_mean_error = float(mean_error_time[-1])
    final_max_error = float(jnp.max(errors[-1]))
    time_averaged_mean_error = float(jnp.mean(mean_error_time))

    rmse_time = jnp.sqrt(jnp.mean(errors ** 2, axis=1))
    final_rmse = float(rmse_time[-1])
    time_averaged_rmse = float(jnp.mean(rmse_time))

    # Settling time (first time mean error <= threshold)
    below = jnp.where(mean_error_time <= threshold)[0]
    settling_time = float(below[0] * dt) if below.size > 0 else None

    # Formation spread at final time: RMS distance to centroid
    final_positions = history[-1]
    centroid = jnp.mean(final_positions, axis=0)
    spread = float(jnp.sqrt(jnp.mean(jnp.sum((final_positions - centroid) ** 2, axis=1))))

    metrics = {
        "final_mean_error": final_mean_error,
        "final_max_error": final_max_error,
        "time_averaged_mean_error": time_averaged_mean_error,
        "final_rmse": final_rmse,
        "time_averaged_rmse": time_averaged_rmse,
        "settling_time": settling_time,
        "formation_spread": spread,
    }

    return metrics, jax.device_get(mean_error_time), jax.device_get(errors)


def format_val(v):
    return "N/A" if v is None else f"{v:0.4f}"


def print_comparison(metrics_a, metrics_b, label_a="Feedforward", label_b="Baseline"):
    keys = list(metrics_a.keys())
    print(f"{'Metric':40s} {label_a:>14s} {label_b:>14s}")
    for k in keys:
        a = metrics_a[k]
        b = metrics_b[k]
        print(f"{k:40s} {format_val(a):>14s} {format_val(b):>14s}")


def compare_and_plot(seed=42, dt=0.05, num_steps=800, plot=True, **sim_kwargs):
    # Run both strategies with the same seed (consistent randomness)
    hist_ff, lead_ff = run_simulation(seed=seed, dt=dt, num_steps=num_steps, feedforward=True, **sim_kwargs)
    plot_results(hist_ff, lead_ff, save_path="results/feedforward_rendezvous.pdf")
    video_plot(hist_ff, lead_ff, save_path="results/feedforward_trajectory.gif", fps=20)
    

    hist_bl, lead_bl = run_simulation(seed=seed, dt=dt, num_steps=num_steps, feedforward=False, **sim_kwargs)
    plot_results(hist_bl, lead_bl, save_path="results/baseline_rendezvous.pdf")
    video_plot(hist_bl, lead_bl, save_path="results/baseline_trajectory.gif", fps=20)

    metrics_ff, mean_err_ff, _ = compute_metrics(hist_ff, lead_ff, dt)
    metrics_bl, mean_err_bl, _ = compute_metrics(hist_bl, lead_bl, dt)

    print("\nStrategy comparison metrics:\n")
    print_comparison(metrics_ff, metrics_bl, label_a="Feedforward", label_b="Baseline")

    if plot:
        T = mean_err_ff.shape[0]
        comparison_plot(T, mean_err_ff, mean_err_bl, dt, save_path="results/error_comparison.pdf")

    return metrics_ff, metrics_bl


if __name__ == "__main__":
    compare_and_plot(seed=42, dt=0.05, num_steps=800, plot=True)
