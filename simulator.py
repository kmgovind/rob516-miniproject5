import jax
import jax.numpy as jnp
from jax import jit, random

"""simulator.py

Provides reusable simulation functions and a `run_simulation` entrypoint
that returns (history, leader_history). The module is importable from
small runtime scripts that set parameters and plot results.
"""


def create_pinning_matrix(num_agents, pinning_indices=(0, 1, 2)):
    g_values = jnp.array([1.0 if i in pinning_indices else 0.0 for i in range(num_agents)])
    return jnp.diag(g_values)


@jit(static_argnums=(1,))
def get_switching_laplacian(key, num_agents, comm_loss_prob):
    adj = jnp.ones((num_agents, num_agents)) - jnp.eye(num_agents)
    mask = random.bernoulli(key, p=1.0 - comm_loss_prob, shape=(num_agents, num_agents))
    adj_masked = adj * mask
    degree = jnp.diag(jnp.sum(adj_masked, axis=1))
    return degree - adj_masked


@jit
def dynamics_step(pos, leader_pos, leader_vel, L, G, dt, k_gain):
    consensus_force = -(L @ pos + G @ (pos - leader_pos))
    dx = leader_vel + k_gain * consensus_force
    return pos + dx * dt


@jit
def dynamics_step_baseline(pos, leader_pos, L, G, dt, k_gain):
    consensus_force = -(L @ pos + G @ (pos - leader_pos))
    dx = k_gain * consensus_force
    return pos + dx * dt


def leader_state(t, radius=5.0, omega=0.2):
    pos = jnp.array([radius * jnp.cos(omega * t), radius * jnp.sin(omega * t)])
    vel = jnp.array([-radius * omega * jnp.sin(omega * t), radius * omega * jnp.cos(omega * t)])
    return pos, vel


def run_simulation(
    num_agents=10,
    dim=2,
    dt=0.05,
    num_steps=800,
    comm_loss_prob=0.4,
    k_gain=1.5,
    seed=42,
    pinning_indices=(0, 1, 2),
    feedforward=True,
    init_pos=None,
    leader_radius=5.0,
    leader_omega=0.2,
):
    """Run the multi-agent rendezvous simulation.

    Returns:
      history: jnp.array shape (num_steps+1, num_agents, dim)
      leader_history: jnp.array shape (num_steps, dim)
    """
    key = random.PRNGKey(seed)

    if init_pos is None:
        key, subkey = random.split(key)
        pos = random.uniform(subkey, shape=(num_agents, dim), minval=-10, maxval=10)
    else:
        pos = jnp.array(init_pos)

    G = create_pinning_matrix(num_agents, pinning_indices)

    history = [pos]
    leader_history = []

    for i in range(num_steps):
        key, subkey = random.split(key)
        t = i * dt
        leader_pos, leader_vel = leader_state(t, radius=leader_radius, omega=leader_omega)
        L = get_switching_laplacian(subkey, num_agents, comm_loss_prob)

        if feedforward:
            pos = dynamics_step(pos, leader_pos, leader_vel, L, G, dt, k_gain)
        else:
            pos = dynamics_step_baseline(pos, leader_pos, L, G, dt, k_gain)

        history.append(pos)
        leader_history.append(leader_pos)

    return jnp.array(history), jnp.array(leader_history)
