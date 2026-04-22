import matplotlib.pyplot as plt
import numpy as np

# IEEE Publication Settings
plt.rcParams.update({
    "text.usetex": False,        # Set to True if you have LaTeX installed on your system
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,             # Match body text size
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.constrained_layout.use": True
})

# Standard IEEE Column Widths (inches)
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.16


def plot_results(history, leader_history, save_path=None):
    # Use SINGLE_COLUMN width for a trajectory plot to save space
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))
    
    # Plot Leader Path with a subtle, distinct style
    ax.plot(leader_history[:, 0], leader_history[:, 1], 
            color='black', linestyle='--', alpha=0.7, label="USV Path")
    
    # Final Leader Position
    ax.scatter(leader_history[-1, 0], leader_history[-1, 1], 
               color='tab:red', marker='X', s=60, zorder=5, label="Final USV Pos")

    num_agents = history.shape[1]
    # Use a colormap for drones so they are distinguishable but cohesive
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_agents))

    for i in range(num_agents):
        # Path
        ax.plot(history[:, i, 0], history[:, i, 1], color=colors[i], alpha=0.4, linewidth=0.8)
        # Final Position
        ax.scatter(history[-1, i, 0], history[-1, i, 1], color=colors[i], s=10, zorder=4)

    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    ax.legend(frameon=True, loc='upper right')
    ax.set_aspect('equal')
    
    if save_path:
        # Save as PDF for vector quality (best for LaTeX/IEEE)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Saved publication-quality plot to {save_path}")
    plt.show()


def comparison_plot(T, mean_err_ff, mean_err_bl, dt, save_path='error_comparison.pdf'):
    times = np.arange(T) * dt
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.5)) # Shorter height for error plots
    
    ax.plot(times, mean_err_ff, color='tab:blue', linestyle='-', label="Proposed (FF)")
    ax.plot(times, mean_err_bl, color='tab:red', linestyle='--', label="Baseline")
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean Tracking Error [m]")
    ax.set_xlim(0, times[-1])
    ax.set_ylim(bottom=0)
    
    ax.legend(frameon=True)
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Saved publication-quality comparison to {save_path}")
    plt.show()