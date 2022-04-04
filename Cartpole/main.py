import gym
import torch
from spinup import ppo_pytorch as PPO
from spinup.utils.test_policy import load_policy_and_env, run_policy

env = lambda : gym.make('CartPole-v0')

ac_kwargs = dict(hidden_sizes=[64,64], activation = torch.nn.ReLU)
logger_kwargs = dict(output_dir='log', exp_name='car')

PPO(env, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs,
    steps_per_epoch=5000, epochs=100)

_, get_action = load_policy_and_env('log')
env_test = gym.make('CartPole-v0')
run_policy(env_test, get_action)