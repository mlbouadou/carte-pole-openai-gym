# CartPole RL Experiments

My first reinforcement learning project, using PyTorch and Gymnasium's CartPole environment.

This initial version provides a reusable foundation for trying different ideas and comparing solutions through Jupyter notebooks.

## Current implementation

- Customizable neural networks: choose the number of hidden layers, their sizes, and each layer's activation function.
- Configurable hidden-layer and output-layer biases.
- REINFORCE training with undiscounted reward-to-go.
- Configurable training batches and optimizer.
- Evaluation utilities for measuring episode rewards and consistency.

The initial architecture uses tanh activations without biases to enforce left-right symmetry.

## Project structure

- src/model.py: configurable policy network.
- src/agent.py: action sampling and action log-probabilities.
- src/train.py: experience collection, returns, loss, and weight updates.
- src/evaluate.py: evaluation without weight updates.
- notebooks/: experiment configuration and analysis.

## Status

This is an initial experimental foundation, not a validated solution.

The evaluation implementation is provisional and will likely change as I refine the comparison methodology.

The goal is to use this project to explore ideas, document experiments, and compare their performance and trade-offs.