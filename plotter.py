import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def video_plot(history, leader_history, save_path='trajectory_video.gif', fps=20):
    # Convert to NumPy
    history_np = np.array(history)
    leader_history_np = np.array(leader_history)
    
    # Ensure both arrays have the same length to avoid IndexErrors
    num_frames = min(history_np.shape[0], leader_history_np.shape[0])
    
    fig, ax = plt.subplots(figsize=(5, 5))
    num_agents = history_np.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_agents))
    
    ax.plot(leader_history_np[:, 0], leader_history_np[:, 1], 
            color='black', linestyle='--', alpha=0.3, label="USV Path")
    
    leader_dot, = ax.plot([], [], color='tab:red', marker='X', markersize=10, ls='', zorder=5)
    agent_dots, = ax.plot([], [], color='blue', marker='o', ls='', markersize=4, zorder=4)
    agent_lines = [ax.plot([], [], color=colors[i], alpha=0.4, linewidth=0.8)[0] for i in range(num_agents)]

    def init():
        all_x = np.concatenate([history_np[:,:,0].flatten(), leader_history_np[:,0]])
        all_y = np.concatenate([history_np[:,:,1].flatten(), leader_history_np[:,1]])
        ax.set_xlim(np.min(all_x) - 1, np.max(all_x) + 1)
        ax.set_ylim(np.min(all_y) - 1, np.max(all_y) + 1)
        ax.set_aspect('equal')
        return [leader_dot, agent_dots] + agent_lines

    def update(frame):
        # Double check bounds inside the loop just in case
        if frame >= num_frames:
            return [leader_dot, agent_dots] + agent_lines
            
        leader_dot.set_data([leader_history_np[frame, 0]], [leader_history_np[frame, 1]])
        agent_dots.set_data(history_np[frame, :, 0], history_np[frame, :, 1])
        
        for i in range(num_agents):
            agent_lines[i].set_data(history_np[:frame+1, i, 0], history_np[:frame+1, i, 1])
        return [leader_dot, agent_dots] + agent_lines

    # Explicitly set frames to num_frames (which is 800, indices 0-799)
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, 
                         blit=True, interval=1000/fps)
    
    print(f"Starting render of {num_frames} frames...")
    anim.save(save_path, writer='pillow', fps=fps)
    print(f"Saved trajectory video to {save_path}")
    plt.close(fig)



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