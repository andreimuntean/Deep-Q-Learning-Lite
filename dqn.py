"""Defines a deep Q-network architecture.

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
    def __init__(self,
                 sess,
                 num_features,
                 num_hidden_units,
                 num_actions,
                 initial_learning_rate,
                 learning_rate_decay_factor,
                 learning_rate_decay_frequency,
                 max_gradient_norm):
        """Creates a deep Q-network that is implemented using a single hidden layer.

        Args:
            sess: The associated TensorFlow session.
            num_features: Number of features in the input vector.
            num_hidden_units: Number of units in the hidden layer.
            num_actions: Number of possible actions. Represents the size of the output layer.
            initial_learning_rate: Initial speed with which the network learns from new examples.
            learning_rate_decay_factor: The value with which the learning rate is multiplied when it
                decays.
            learning_rate_decay_frequency: The frequency (measured in training steps) at which the
                learning rate is reduced.
            max_gradient_norm: Maximum value allowed for the L2-norms of gradients. Gradients with
                norms that would otherwise surpass this value are scaled down.
        """

        self.sess = sess
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

        # Compare with the observed action value.
        self.observed_action_value = tf.placeholder(
            tf.float32, [None], name='Observed_Action_Value')

        # Compute the loss function and regularize it by clipping the norm of its gradients.
        loss = tf.nn.l2_loss(self.estimated_action_value - self.observed_action_value)
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # Perform gradient descent.
        grads_and_vars = list(zip(clipped_gradients, tf.trainable_variables()))
        self.global_step = tf.Variable(tf.constant(0, tf.int64), False, name='Global_Step')
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                        self.global_step,
                                                        learning_rate_decay_frequency,
                                                        learning_rate_decay_factor,
                                                        staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            grads_and_vars, self.global_step)

    def eval_optimal_action(self, state):
        """Estimates the optimal action for the specified state.

        Args:
            state: A state. Can be batched into multiple states.
        """

        return self.sess.run(self.optimal_action, feed_dict={self.x: state})

    def eval_optimal_action_value(self, state):
        """Estimates the optimal action value for the specified state.

        Args:
            state: A state. Can be batched into multiple states.
        """

        return self.sess.run(self.optimal_action_value, feed_dict={self.x: state})

    def eval_Q(self, state, action):
        """Evaluates the utility of the specified action for the specified state.

        Args:
            state: A state. Can be batched into multiple states.
            action: An action. Can be batched into multiple actions.
        """

        return self.sess.run(self.estimated_action_value, feed_dict={self.x: state,
                                                                     self.action: action})

    def train(self, state, action, observed_action_value):
        """Learns by performing one step of gradient descent.

        Args:
            state: A state. Can be batched into multiple states.
            action: An action. Can be batched into multiple actions.
            observed_action_value: An observed action value (the ground truth).
        """

        self.sess.run(self.train_step, feed_dict={
            self.x: state, self.action: action, self.observed_action_value: observed_action_value})
