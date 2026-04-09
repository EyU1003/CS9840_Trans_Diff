# Differentiable User Equilibrium Embedded Diffusion Model for Urban Transportation Network Design

A diffusion-based framework for Transportation Network Design Problems (TNDP) with an embedded differentiable User Equilibrium (UE) layer.

## Overview

This project studies how generative models can be applied to transportation network design under realistic traffic behavior. Traditional TNDP is formulated as a bi-level optimization problem where the upper level optimizes the network design and the lower level solves User Equilibrium traffic assignment. Since User Equilibrium is typically treated as a black-box solver, gradients cannot flow from travel outcomes back to the network design.

To address this limitation, this project integrates a diffusion model with a differentiable UE layer, allowing end-to-end optimization of transportation networks.

## Framework

The overall pipeline is:

<p align="center">
  <img src="plots/frame.png" width="700"/>
</p>

```text
(G0, Q, C) -> Diffusion Model -> Generated Network G' -> UE Solver -> Travel Time Cost
```

Where:

* `G0` = initial network structure
* `Q` = travel demand
* `C` = design constraints such as budget
* `G'` = generated network design
* `f*` = equilibrium traffic flow

The forward pass computes:

```text
design -> equilibrium flow -> travel cost
```

The backward pass propagates gradients through the differentiable UE solver back into the diffusion model.

## Main Components

### Diffusion-Based Network Generator

* Learns feasible network expansions
* Conditioned on demand level and budget
* Represents edge capacities as continuous variables
* Generates improved network structures under different traffic scenarios

### Differentiable User Equilibrium Layer

* Computes equilibrium traffic flow for a generated network
* Allows gradients to pass through the traffic assignment process
* Connects network design decisions with downstream congestion and travel time

### Objective Function

```text
Loss = Total Travel Time + λ1 * Regularization + λ2 * Budget Penalty
```

This objective encourages low travel time while enforcing feasible network expansion.

## Repository Structure

```text
CS9840_Trans_Diff/
├── configs/
│   └── configs.py
├── models/
│   ├── model.py
│   ├── network.py
│   └── saved/
│       └── model.pt
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   ├── logs/
│   ├── plots/
│   └── checkpoints/
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Code

Train the model:

```bash
python scripts/train.py
```

Evaluate the model:

```bash
python scripts/evaluate.py
```

## Key Idea

The key contribution of this project is making the User Equilibrium layer differentiable so that traffic outcomes directly influence the learned network design. This enables transportation-aware generative modeling under realistic congestion behavior.
