"""Defines the architecture of a deep Q-network.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import math
import tensorflow as tf


def _fully_connected_layer(x, shape, bias_shape, activation_fn):
    if len(shape) != 2:
        raise ValueError('Shape "{}" is invalid. Must have length 2.'.format(shape))

    maxval = 1 / math.sqrt(shape[0] + shape[1])
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.constant(0.1, tf.float32, bias_shape), name='Bias')

    return activation_fn(tf.matmul(x, W) + b)


class DeepQNetwork():
    """A neural network that learns the Q (action value) function."""

    def __init__(self, num_features, num_hidden_units, num_actions):
        """Creates a deep Q-network that is implemented using a single hidden layer.

        Args:
            num_features: Number of features in the input vector.
            num_hidden_units: Number of units in the hidden layer.
            num_actions: Number of possible actions. Represents the size of the output layer.
        """

        self.x = tf.placeholder(tf.float32, [None, num_features], name='Input_States')

        with tf.name_scope('Fully_Connected_Layer_1'):
            h_fc = _fully_connected_layer(
                self.x, [num_features, num_hidden_units], [num_hidden_units], tf.nn.relu)

        with tf.name_scope('Fully_Connected_Layer_2'):
            # Use a single shared bias for each action.
            self.Q = _fully_connected_layer(h_fc, [num_hidden_units, num_actions], [1], tf.identity)

        # Estimate the optimal action and its expected value.
        self.optimal_action = tf.squeeze(tf.argmax(self.Q, 1, name='Optimal_Action'))
        self.optimal_action_value = tf.squeeze(tf.reduce_max(self.Q, 1))

        # Estimate the value of the specified action.
        self.action = tf.placeholder(tf.uint8, name='Action')
        one_hot_action = tf.one_hot(self.action, num_actions)
        self.estimated_action_value = tf.reduce_sum(self.Q * one_hot_action, 1)

    def get_action_value(self, state, action):
        """Estimates the value of the specified action for the specified state.

        Args:
            state: State of the environment. Can be batched into multiple states.
            action: A valid action. Can be batched into multiple actions.
        """

        sess = tf.get_default_session()
        return sess.run(self.estimated_action_value, {self.x: state, self.action: action})

    def get_optimal_action_value(self, state):
        """Estimates the optimal action value for the specified state.

        Args:
            state: State of the environment. Can be batched into multiple states.
        """

        sess = tf.get_default_session()
        return sess.run(self.optimal_action_value, {self.x: state})

    def get_optimal_action(self, state):
        """Estimates the optimal action for the specified state.

        Args:
            state: State of the environment. Can be batched into multiple states.
        """

        sess = tf.get_default_session()
        return sess.run(self.optimal_action, {self.x: state})
