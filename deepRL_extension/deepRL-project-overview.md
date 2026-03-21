# Deep RL Extension: Multi-Agent Learning Dynamics

## Overview

This extension investigates whether the complex learning dynamics observed in tabular multi-agent Q-learning (transient cooperation, metastability, oscillations) persist when using deep reinforcement learning algorithms.

**Core hypothesis:** The replay buffer used in standard deep RL dampens or eliminates the complex dynamics by breaking the tight feedback loop between simultaneously adapting agents.

## Research Questions

1. Does DQN **without replay** exhibit similar complex dynamics to tabular Q-learning?
2. Does enabling the replay buffer dampen/eliminate these dynamics?
3. Where does PPO (on-policy, batched updates) fall on this spectrum?
4. How does the discount factor γ affect dynamics in each case?

## Experimental Conditions

| Condition | Algorithm | Replay | Update Style | Expected Dynamics |
|-----------|-----------|--------|--------------|-------------------|
| 1 | DQN | Off | Incremental | Complex (like tabular) |
| 2 | DQN | On | Batched, decorrelated | Damped/simplified |
| 3 | PPO | N/A | Batched, on-policy | Intermediate? |

**Baseline comparison:** Tabular Q-learning from main paper

## Environment Specification

- **Game:** 2-player normal-form games (default: Prisoner's Dilemma)
- **State space:** Single state (stateless environment)
- **Action space:** 2 actions per agent (cooperate/defect)
- **Reward matrix:** Modular, supports PD, Matching Pennies, Stag Hunt, Bach-Stravinsky

### Default Parameters (matching main paper)

| Parameter | Value |
|-----------|-------|
| Temperature (T) | 1.0 |
| Learning rate (α) | 0.01 |
| Discount factor (γ) | Variable (0.0, 0.5, 0.8, 0.95) |
| Episodes | 100,000 - 1,000,000 |

## Directory Structure

```
deepRL_extension/
├── deepRL-project-overview.md    # This file
├── requirements.txt              # Additional dependencies (torch)
├── agents/
│   ├── __init__.py
│   ├── base.py                   # BaseAgent interface
│   ├── dqn.py                    # SimpleDQN (toggleable replay)
│   └── ppo.py                    # SimplePPO
├── environment.py                # TwoPlayerGame environment
├── replay_buffer.py              # ReplayBuffer class
├── simulation.py                 # MultiAgentSimulation class
└── utils.py                      # Logging, plotting helpers

# Notebook in root (alongside other PaperCompanion notebooks)
PaperCompanionDQN.ipynb           # Main experiment notebook
```

## Implementation Plan

If possible, look at the main code in the paper as a reference. But this code was written by me 2 years ago, so it is probably in many ways of flawed quality. Try to mimick the style where it makes sense but do not sacrify efficiency or best practices by doing so.

### Phase 1: Core Infrastructure
- [ ] `environment.py` — TwoPlayerGame class
  - Modular reward matrices (reuse from agent_game_sim.py)
  - Step function: both agents act, return rewards
  - Track cooperation probabilities over time

- [ ] `agents/base.py` — BaseAgent interface
  - `get_action_probabilities()` — for logging dynamics
  - `choose_action(temperature)` — Boltzmann sampling
  - `update(action, reward, gamma)` — learning step
  - `get_q_values()` or equivalent — for analysis

- [ ] `replay_buffer.py` — Simple replay buffer
  - Store (action, reward) transitions
  - Sample random batches
  - Toggle on/off functionality

### Phase 2: Agents
- [ ] `agents/dqn.py` — SimpleDQN
  - Small MLP: input (dummy state) → hidden → 2 Q-values
  - Boltzmann action selection over Q-values
  - TD learning update
  - Optional replay buffer integration

- [ ] `agents/ppo.py` — SimplePPO
  - Actor-critic network: shared hidden → policy logits + value
  - Sample actions from policy directly
  - PPO-clip objective for policy update
  - Value function loss for critic

### Phase 3: Simulation & Analysis
- [ ] `simulation.py` — MultiAgentSimulation
  - Training loop for two independent agents
  - Log: cooperation probabilities, Q-values/policies over time
  - Support for parameter schedules (if needed)
  - Save results in format compatible with existing data/

- [ ] `PaperCompanionDQN.ipynb` — Experiments
  - Run all three conditions
  - Compare with tabular baseline
  - Visualize dynamics (time evolution, phase portraits)
  - Analyze effect of discount factor

## Technical Notes

### Why Custom Implementation (not SB3)?

1. **Incremental updates:** SB3 requires replay buffer; we need to disable it
2. **Simplicity:** Stateless environment with 2 actions doesn't need full SB3 machinery
3. **Control:** Need direct access to Q-values/policies for dynamics analysis
4. **Comparability:** Match update rule to tabular case as closely as possible

### Neural Network Architecture

For this simple case (single state, 2 actions):

```
DQN:
  Input: [1] (dummy state, e.g., constant 1.0)
  Hidden: 32 units, ReLU
  Output: [2] Q-values (Q_cooperate, Q_defect)

PPO:
  Input: [1] (dummy state)
  Hidden: 32 units, ReLU
  Policy head: [2] logits
  Value head: [1] scalar
```

### Key Differences from Tabular

| Aspect | Tabular Q-learning | Neural DQN |
|--------|-------------------|------------|
| Q-value storage | Lookup table | Network weights |
| Update | Direct: Q += α·δ | Gradient descent on MSE loss |
| Action coupling | Independent Q(a) | Shared weights affect both outputs |
| Generalization | None | Implicit (though minimal here) |

## Expected Outcomes

1. **DQN (no replay):** Qualitatively similar dynamics to tabular, possibly with different fixed points or stability due to gradient-based updates

2. **DQN (with replay):** Faster convergence to Nash equilibrium, reduced oscillations, loss of transient cooperation phenomena

3. **PPO:** Intermediate behavior — on-policy nature preserves some coupling, but batched updates may smooth dynamics

## Dependencies

```
torch>=2.0
numpy
matplotlib
```

## Hypothesis: Policy Gradient Methods (REINFORCE, PPO)

We hypothesise that policy gradient methods (REINFORCE, PPO) would **not** show the same complex dynamics (oscillations, metastability) observed in Q-learning. The key reason is that the complex dynamics are caused by **bootstrapping** — the self-referential term γ · max(Q) in the Q-learning update:

```
Q(a) ← Q(a) + α [r + γ · max(Q) - Q(a)]
```

With γ > 0, the TD target depends on the current Q-values themselves. Both agents simultaneously chase moving targets that depend on each other's Q-values. This coupled, self-referential feedback is what generates the complex dynamics described by the deterministic model.

**REINFORCE** uses actual sampled returns G_t = Σ γ^k r_{t+k}, not bootstrap estimates. There is no self-referential term — the gradient is computed from observed rewards only. In our stateless single-step game, this reduces to a bandit problem (G = r), and we would expect noisy but relatively monotonic convergence toward defection.

**PPO** also updates the policy directly and clips the surrogate objective, explicitly limiting how much the policy can change per update. Even with GAE (which involves some bootstrapping via the value function), the clipping mechanism would suppress oscillatory dynamics by design.

| Method | Bootstrapping | Tight feedback loop | Expected dynamics (γ=0.8) |
|--------|:---:|:---:|---|
| Tabular Q-learning | Yes (γ max Q) | Yes | Complex oscillations |
| DQN (no replay) | Yes (γ max Q) | Yes | Similar oscillations |
| DQN (with replay) | Yes (γ max Q) | Broken by replay | Damped, near fixed point |
| REINFORCE | No | Partial | Noisy but monotonic convergence |
| PPO | Partial (GAE) | Clipped | Smooth, stable convergence |

In summary, the complex dynamics are a specific property of the **Q-learning update rule with bootstrapping**, not a general feature of multi-agent learning. Policy gradient methods produce fundamentally different coupled update equations and would likely show qualitatively different (smoother) dynamics.

## References

- Main paper: "Deterministic Model of Incremental Multi-Agent Boltzmann Q-Learning" (Goll, Heitzig, Barfuss 2024)
- DQN: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
