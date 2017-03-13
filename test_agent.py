"""Tests a trained agent's ability to maximize its score in OpenAI Gym environments."""

import argparse
import agent
import environment
import logging
import tensorflow as tf

from gym import wrappers


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PARSER = argparse.ArgumentParser()


PARSER.add_argument('env_name', help='name of the OpenAI Gym environment that will be played')

PARSER.add_argument('load_path', help='loads the trained model from the specified path')

PARSER.add_argument('--save_path',
                    metavar='PATH',
                    default=None,
                    help='path where to save experiments and videos')

PARSER.add_argument('--render',
                    help='determines whether to display the game screen of the agent',
                    dest='render',
                    action='store_true')

PARSER.set_defaults(render=False)

PARSER.add_argument('--num_episodes',
                    help='number of episodes to play',
                    type=int,
                    default=10)

PARSER.add_argument('--action_space',
                    nargs='+',
                    help='restricts the number of possible actions',
                    type=int)

PARSER.add_argument('--max_episode_length',
                    metavar='TIME STEPS',
                    help='maximum number of time steps per episode',
                    type=int,
                    default=5000)

PARSER.add_argument('--num_hidden_units',
                    metavar='NEURONS',
                    help='number of units in the hidden layer of the network',
                    type=int,
                    default=40)


def main(args):
    """Loads a trained agent that maximizes its score in OpenAI Gym environments."""

    env = environment.EnvironmentWrapper(
        args.env_name, args.max_episode_length, 100, args.action_space)

    if args.save_path:
        env.gym_env = wrappers.Monitor(env.gym_env, args.save_path)

    env.reset()

    with tf.Session() as sess:
        player = agent.TestOnlyAgent(env, args.num_hidden_units)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.load_path)

        for _ in range(args.num_episodes):
            episode_reward = 0

            for t in range(args.max_episode_length):
                state = env.get_state()
                action = player.get_action(state)
                reward = env.step(action)
                episode_reward += reward

                if args.render:
                    env.render()

                if env.done:
                    break

            LOGGER.info('Episode finished after %d time steps. Reward: %d.', t + 1, episode_reward)


if __name__ == '__main__':
    main(PARSER.parse_args())
