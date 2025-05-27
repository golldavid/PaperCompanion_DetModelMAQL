# PaperCompanion: Deterministic Model of Incremental Multi-Agent Boltzmann Q-Learning

This repository contains a collection of Jupyter notebooks and Python modules to replicate the results presented in "Deterministic Model of Incremental Multi-Agent Boltzmann Q-Learning: Transient Cooperation, Metastability, and Oscillations" (D. Goll, J. Heitzig, W. Barfuss, 2024, ArXiv).
The notebooks can be used for simulating, analyzing, and visualizing learning dynamics in repeated games, with a focus on the Prisoner's Dilemma. 

## Repository Structure

- `PaperCompanion1_I.ipynb`, `PaperCompanion1_II.ipynb`, ..., `PaperCompanion5.ipynb`:  
  Jupyter notebooks for generating figures and running experiments as described in the paper. Each notebook corresponds to a specific figure.
- `agent_game_sim.py`:  
  Core Python module containing classes and functions for agent-based simulations, Q-learning, and game setup.
- `requirements.txt`:  
  List of required Python packages.
- `data/`:  
  Directory for simulation data and intermediate results.
- `PaperFigures/`:  
  Output directory for generated figures.

## Getting Started

### Prerequisites

- Python 3.10 or later

### Installation

1. Clone this repository:
    ```sh
    git clone <repository-url>
    cd PaperCompanion_DetModelMAQL
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Open any of the `PaperCompanion*.ipynb` notebooks in JupyterLab or VS Code.
2. Run the cells to reproduce the figures and analyses from the paper.
3. Generated figures will be saved in the `PaperFigures/` directory.

#### Example

To reproduce the deterministic learning trajectories in policy space (Figure 3):

- Open `PaperCompanion3.ipynb`
- Run all cells
- The resulting figures will be saved in `PaperFigures/`

## Data

- Simulation data is saved and loaded from the `data/` directory.
- If `load_data` is set to `False` in a notebook, new simulations will be run and data will be generated.

## Project Highlights

- **Deterministic and Stochastic Q-Learning**:  
  Simulate both deterministic and stochastic learning processes.
- **Flexible Experimentation**:  
  Easily modify learning rates, discount factors, temperatures, and initial conditions.
