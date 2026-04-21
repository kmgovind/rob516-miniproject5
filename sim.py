import jax
import jax.numpy as jnp
from jax import vmap, jit, random
import matplotlib.pyplot as plt

# --- Hyperparameters ---
num_agents = 10
dim = 2
dt = 0.05
num_steps = 800  # Longer simulation to show settling
comm_loss_prob = 0.4 
k_gain = 1.5      # Higher gain reduces the remaining error

# --- Initialization ---
key = random.PRNGKey(42)
key, subkey = random.split(key)
pos = random.uniform(subkey, shape=(num_agents, dim), minval=-10, maxval=10)

# Pinning: Only drones 0, 1, and 2 can see the USV
g_values = jnp.array([1.0, 1.0, 1.0] + [0.0] * (num_agents - 3))
G = jnp.diag(g_values)

@jit(static_argnums=(1,))
def get_switching_laplacian(key, num_agents):
    adj = jnp.ones((num_agents, num_agents)) - jnp.eye(num_agents)
    mask = random.bernoulli(key, p=1.0 - comm_loss_prob, shape=(num_agents, num_agents))
    adj_masked = adj * mask
    degree = jnp.diag(jnp.sum(adj_masked, axis=1))
    return degree - adj_masked

@jit
def dynamics_step(pos, leader_pos, leader_vel, L, G, dt):
    """
    Updated Dynamics: dx = leader_vel - k*(L*x + G*(x - x_leader))
    The 'leader_vel' is the feedforward term that eliminates tracking lag.
    """
    # Standard consensus + pinning
    consensus_force = -(L @ pos + G @ (pos - leader_pos))
    
    # Dynamics = Feedforward + Proportional Correction
    dx = leader_vel + k_gain * consensus_force
    return pos + dx * dt

# --- Simulation Loop ---
history, leader_history = [pos], []

for i in range(num_steps):
    key, subkey = random.split(key)
    t = i * dt
    
    # Leader Path: r=5, omega=0.2
    leader_pos = jnp.array([5 * jnp.cos(0.2 * t), 5 * jnp.sin(0.2 * t)])
    # Derivative of position = velocity (Feedforward term)
    leader_vel = jnp.array([-1.0 * jnp.sin(0.2 * t), 1.0 * jnp.cos(0.2 * t)])
    
    L = get_switching_laplacian(subkey, num_agents)
    pos = dynamics_step(pos, leader_pos, leader_vel, L, G, dt)
    
    history.append(pos)
    leader_history.append(leader_pos)

# --- Plotting ---
history = jnp.array(history)
leader_history = jnp.array(leader_history)

plt.figure(figsize=(10, 10))
plt.plot(leader_history[:, 0], leader_history[:, 1], 'k--', label="USV Path")
plt.scatter(leader_history[-1, 0], leader_history[-1, 1], c='red', marker='X', s=200, label="Final USV Pos")

for i in range(num_agents):
    plt.plot(history[:, i, 0], history[:, i, 1], alpha=0.4)
    plt.scatter(history[-1, i, 0], history[-1, i, 1], s=20, edgecolors='black')

plt.title(f"Rendezvous with Feedforward (Eliminating Tracking Lag)")
plt.legend()
plt.grid(True, linestyle=':')
plt.axis('equal')
plt.show()