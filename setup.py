"""Installs the modules required to run train_agent.py and test_agent.py."""

from setuptools import setup


setup(
    name='Deep Q-Network Lite',
    version='1.0.0',
    url='https://github.com/andreimuntean/Deep-Q-Learning-Lite',
    description='Deep reinforcement learning for environments with small state spaces.',
    author='Andrei Muntean',
    keywords='deep learning machine reinforcement neural network q-network dqn openai',
    install_requires=['gym', 'numpy', 'tensorflow']
)
