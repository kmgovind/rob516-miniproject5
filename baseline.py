from simulator import run_simulation
import jax
import matplotlib.pyplot as plt


if __name__ == "__main__":
    history, leader_history = run_simulation(feedforward=False)
    history = jax.device_get(history)
    leader_history = jax.device_get(leader_history)

    plt.figure(figsize=(10, 10))
    plt.plot(leader_history[:, 0], leader_history[:, 1], 'k--', label="USV Path")
    plt.scatter(leader_history[-1, 0], leader_history[-1, 1], c='red', marker='X', s=200, label="Final USV Pos")

    for i in range(history.shape[1]):
        plt.plot(history[:, i, 0], history[:, i, 1], alpha=0.4)
        plt.scatter(history[-1, i, 0], history[-1, i, 1], s=20, edgecolors='black')

    plt.title("Baseline Rendezvous (Consensus Only, No Feedforward)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.axis('equal')
    plt.show()