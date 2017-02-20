"""Augments OpenAI Gym environments with features like experience replay.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import gym
import numpy as np
import random
import time


class EnvironmentWrapper:
    """Wraps over an OpenAI Gym environment and provides experience replay."""

    def __init__(self, env_name, max_episode_length, replay_memory_capacity, action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym environment.
            max_episode_length: Maximum number of time steps per episode. When this number of time
                steps is reached, the episode terminates early.
            replay_memory_capacity: Number of experiences remembered. Conceptually, an experience is
                a (state, action, reward, next_state, done) tuple. The replay memory is sampled by
                the agent during training.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
        """

        self.gym_env = gym.make(env_name)
        self.max_episode_length = max_episode_length
        self.replay_memory_capacity = replay_memory_capacity
        self.num_features = self.gym_env.observation_space.shape[0]
        self.reset()

        if action_space:
            self.action_space = list(action_space)
        else:
            self.action_space = list(range(self.gym_env.action_space.n))

        self.num_actions = len(self.action_space)

        # Create replay memory. Arrays are used instead of double-ended queues for faster indexing.
        self.num_exp = 0
        self.actions = np.empty(replay_memory_capacity, np.uint8)
        self.rewards = np.empty(replay_memory_capacity, np.int8)
        self.ongoing = np.empty(replay_memory_capacity, np.bool)

        # Used for computing both 'current' and 'next' states.
        self.observations = np.empty([replay_memory_capacity + 1, self.num_features], np.float32)

        # Initialize the first state.
        self.observations[0] = self.gym_env.reset()

        # Initialize the first experience by performing one more random action.
        self.step(self.sample_action())

    def reset(self):
        """Resets the environment."""

        self.done = False
        self.gym_env.reset()
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_start_time = time.time()
        self.episode_run_time = 0
        self.fps = 0

    def step(self, action):
        """Performs the specified action.

        Returns:
            The reward.

        Raises:
            ValueError: If the action is not valid.
        """

        if self.done:
            self.reset()

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        observation, reward, self.done, _ = self.gym_env.step(action)

        self.episode_reward += reward
        self.episode_length += 1
        self.episode_run_time = time.time() - self.episode_start_time
        self.fps = 0 if self.episode_run_time == 0 else self.episode_length / self.episode_run_time

        if self.episode_length == self.max_episode_length:
            self.done = True

        # Remember this experience.
        self.actions[self.num_exp] = action
        self.rewards[self.num_exp] = reward
        self.ongoing[self.num_exp] = not self.done
        self.observations[self.num_exp + 1] = observation
        self.num_exp += 1

        if self.num_exp == self.replay_memory_capacity:
            # Free up space by deleting half of the oldest experiences.
            mid = int(self.num_exp / 2)
            end = 2 * mid

            self.num_exp = mid
            self.actions[:mid] = self.actions[mid:end]
            self.rewards[:mid] = self.rewards[mid:end]
            self.ongoing[:mid] = self.ongoing[mid:end]
            self.observations[:mid + 1] = self.observations[mid:end + 1]

        return reward

    def render(self):
        """Draws the environment."""

        self.gym_env.render()

    def sample_action(self):
        """Samples a random action."""

        return random.choice(self.action_space)

    def sample_experiences(self, exp_count):
        """Randomly samples experiences from the replay memory. May contain duplicates.

        Args:
            exp_count: Number of experiences to sample.

        Returns:
            A (states, actions, rewards, next_states, ongoing) tuple. The boolean array, 'ongoing',
            determines whether the 'next_states' are terminal states.
        """

        indexes = np.random.choice(self.num_exp, exp_count)
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        ongoing = self.ongoing[indexes]
        states = self.observations[indexes]
        next_states = self.observations[indexes + 1]

        return states, actions, rewards, next_states, ongoing

    def get_state(self):
        """Gets the current state.

        Returns:
            A tensor with float32 values.
        """

        return np.expand_dims(self.observations[self.num_exp], axis=0)
