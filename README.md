# Deep Q-Learning Lite
Deep reinforcement learning for environments with small state spaces.

## Dependencies
* NumPy
* OpenAI Gym 0.8
* TensorFlow 1.0

## Learning Environment
Uses environments provided by [OpenAI Gym](https://gym.openai.com/).

## Network Architecture
The network has a single hidden layer with 40 rectified linear units. The output layer has as many nodes as there are actions. Each output node represents the expected utility of an action.

## Acknowledgements
Heavily influenced by DeepMind's seminal paper ['Playing Atari with Deep Reinforcement Learning' (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602) and ['Human-level control through deep reinforcement learning' (Mnih et al., 2015)](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).
