# Resilient Rendezvous for Heterogeneous Marine-Aerial Teams Under Communication Uncertainty

Short description
-----------------

This repository contains the simulation code and plotting utilities used for the project "Resilient Rendezvous for Heterogeneous Marine-Aerial Teams Under Communication Uncertainty." The code implements a simple, reproducible simulation pipeline (written with JAX) for evaluating decentralized rendezvous controllers where a surface vehicle (USV) provides a leader trajectory and a team of aerial agents (UAVs) attempt to rendezvous despite intermittent communications.

The codebase includes a baseline consensus-only strategy and a proposed feedforward-enhanced strategy; utilities are provided to compare them, compute simple tracking metrics, and produce publication-quality figures.

Authors
-------

- Kavin M. Govindarajan — PhD Candidate, Robotics, University of Michigan
- Sanyam Mehta — MS Student, Robotics, University of Michigan

Repository
----------

Code and examples are in this repository. If published alongside a paper, please cite the repository and authors (example BibTeX below).

Table of contents
-----------------

- Installation and prerequisites
- Quick start (run examples)
- Reproducing figures and experiments
- Code structure and file descriptions
- Simulation parameters and configuration
- Evaluation metrics and outputs
- Citation and license
- License & acknowledgements

Installation and prerequisites
------------------------------

Requirements

- Python 3.10 or newer (3.10+ recommended)
- pip
- The project pins dependencies in `requirements.txt` (JAX, jaxlib, numpy, matplotlib).

Create and activate a virtual environment

On Windows (PowerShell):

```powershell
python -m venv projvenv
.\\projvenv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

On Windows (CMD):

```cmd
python -m venv projvenv
projvenv\\Scripts\\activate.bat
pip install -r requirements.txt
```

On macOS / Linux (bash):

```bash
python -m venv projvenv
source projvenv/bin/activate
pip install -r requirements.txt
```

Notes

- JAX installation can require platform-specific wheels (GPU vs CPU). If `pip install -r requirements.txt` fails on `jaxlib`, consult the JAX installation documentation for the appropriate wheel for your platform.

Quick start — run the examples
--------------------------------

From the `code/` directory, run one of the convenience scripts:

```bash
python run_sim.py         # Run feedforward strategy and show a trajectory plot
python run_baseline.py    # Run baseline (no feedforward) and show a plot
python strat_comparison.py# Run both strategies, compute metrics, and save PDF figures to results/
```

- `run_sim.py` calls `simulator.run_simulation(feedforward=True)` then plots the result interactively.
- `run_baseline.py` calls `simulator.run_simulation(feedforward=False)` and plots the baseline.
- `strat_comparison.py` runs both strategies with the same PRNG seed, computes comparison metrics, and saves the publication-quality PDFs to the `results/` folder: `results/feedforward_rendezvous.pdf`, `results/baseline_rendezvous.pdf`, and `results/error_comparison.pdf`.

Reproducibility
---------------

- The simulation uses a deterministically-seeded PRNG (`seed=42` by default). To reproduce results exactly, fix the `seed` argument when calling `run_simulation` or executing `strat_comparison.py`.
- The `strat_comparison.py` script saves results to the `results/` directory — ensure that directory exists and is writable.

Code structure and file descriptions
-----------------------------------

- `simulator.py` — Core simulation functions. Exposes `run_simulation(...)` which returns `(history, leader_history)` arrays. The function parameters and defaults are documented in the function signature.
- `plotter.py` — Publication-quality plotting utilities and constants (`SINGLE_COLUMN`, `DOUBLE_COLUMN`) with functions `plot_results(...)` and `comparison_plot(...)` that save vector PDF figures suitable for LaTeX/IEEE workflows.
- `run_sim.py`, `run_baseline.py`, `sim.py`, `baseline.py` — Convenience scripts to run single experiments and display plots interactively.
- `strat_comparison.py` — Runs both the feedforward-enhanced strategy and the baseline, computes metrics (`compute_metrics`) and produces comparison figures saved under `results/`.
- `requirements.txt` — Pinned Python dependencies used for development and reproduction.
- `results/` — Output directory (figures and saved PDFs).

Simulation parameters (defaults)
-------------------------------

The main entrypoint is `simulator.run_simulation(...)`. Important parameters (and defaults) are:

- `num_agents=10` — number of UAV agents
- `dim=2` — environment dimension (2D)
- `dt=0.05` — simulation timestep (s)
- `num_steps=800` — number of simulation steps
- `comm_loss_prob=0.4` — per-edge communication loss probability for switching Laplacian
- `k_gain=1.5` — consensus control gain
- `seed=42` — PRNG seed for reproducibility
- `pinning_indices=(0, 1, 2)` — indices of agents pinned to leader information (USV receivers)
- `feedforward=True` — enable leader feedforward term (proposed method); set to `False` for baseline
- `leader_radius=5.0`, `leader_omega=0.2` — parameters for the leader (USV) circular trajectory

Tuning these parameters lets you explore robustness to communication loss, team size, controller gains, and leader motion.

Evaluation metrics
------------------

`strat_comparison.py` computes the following metrics via `compute_metrics(...)`:

- `final_mean_error` — mean distance to leader at final time
- `final_max_error` — maximum agent distance to leader at final time
- `time_averaged_mean_error` — time-averaged mean error over the run
- `final_rmse` — root-mean-square error at final time
- `time_averaged_rmse` — time-averaged RMSE
- `settling_time` — first time at which the mean error falls below a specified threshold
- `formation_spread` — RMS distance of agents to their centroid at final time

These metrics are printed to the console and can be used to report quantitative comparisons in a paper.

Figures and publication-quality output
-------------------------------------

- `plotter.py` configures Matplotlib for compact, IEEE-style figures (serif font, single/double column widths, PDF export). Use `plot_results(...)` and `comparison_plot(...)` to generate vector-quality PDFs for inclusion in manuscripts.
- By default, files are saved as PDF in `results/` when called from `strat_comparison.py`.

Extending and experiments
-------------------------

- To test new algorithms, implement alternative dynamics or controllers and call into `simulator.run_simulation(...)` from a new script.
- The core simulation uses a switching Laplacian (`get_switching_laplacian`) to model random communication losses; this is a convenient hook to evaluate different communication models.

Citation
--------

If you use this code in published work, please cite the authors and repository. Example BibTeX entry:

```bibtex
@misc{govindarajan2026resilient,
	title = {Resilient Rendezvous for Heterogeneous Marine-Aerial Teams Under Communication Uncertainty},
	author = {Kavin M. Govindarajan and Sanyam Mehta},
	year = {2026},
	howpublished = {\\url{https://github.com/kmgovind/rob516-miniproject5}},
	note = {Code and simulation environment}
}
```

License
-------

No license file is included in this repository by default. For public release, choose and add a license (we recommend `MIT` for code; consider `CC-BY` for paper artifacts). Add a `LICENSE` file at the repository root.

Acknowledgements
----------------

This project was developed as part of the ROB 516 coursework at the University of Michigan. The authors thank collaborators and reviewers for feedback.

Contact
-------

For questions or issues, please open an issue on the repository or contact the authors via the project page.

Useful quick references
----------------------

- Run the comparison script to reproduce figures and metrics:

```bash
python strat_comparison.py
```

- Run a single interactive run (feedforward):

```bash
python run_sim.py
```

---

The file `simulator.py` contains function-level documentation and default parameter values; consult it when extending or reparameterizing experiments.
