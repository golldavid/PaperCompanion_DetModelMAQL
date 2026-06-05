# PaperCompanion: Deterministic Model of Incremental Multi-Agent Boltzmann Q-Learning

Companion code for:
- arXiv preprint (Dez 2024): https://arxiv.org/pdf/2501.00160
- Published paper (Apr 2026): https://doi.org/10.3390/app16073524

This repository contains Jupyter notebooks and Python modules to reproduce the results from the paper.
It focuses on learning dynamics in repeated games such as the Prisoner's Dilemma.

## Repository Structure

- `PaperCompanion1_I.ipynb`, `PaperCompanion1_II.ipynb`, ..., `PaperCompanion6.ipynb`  
  Notebooks for generating the main figures and experiments.
- `PaperCompanionAppendix1.ipynb`, `PaperCompanionAppendix2.ipynb`  
  Additional appendix analyses.
- `agent_game_sim.py`  
  Core simulation logic for agent-based Q-learning and game setup.
- `paper/`  
  Paper-related assets and supplementary material.

## Requirements

- Python 3.10 or later

## Installation

1. Clone this repository:
   ```sh
   git clone <repository-url>
   cd PaperCompanion_DetModelMAQL
   ````

2. Install dependencies:
   ```sh
   pip install .
   ```

If you use `uv`, install from the lockfile:
   ```sh
   uv sync
   ```

Optional deep-RL dependencies:
   ```sh
   pip install ".[deeprl]"
   ```

## Usage

1. Open one of the `PaperCompanion*.ipynb` notebooks in JupyterLab or VS Code.
2. Run the notebook cells to reproduce figures and analyses.
3. Generated figures are saved to `PaperFigures/`.

## Data

- Simulation data is saved and loaded from the `data/` directory.
- If `load_data` is set to `False` in a notebook, new simulations will be run and data will be generated.
