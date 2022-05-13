import gym
import torch
from spinup import ppo_pytorch as ppo
from spinup.utils.test_policy import load_policy_and_env, run_policy

env = lambda: gym.make('InvertedPendulum-v2')  # lambda function prevent duplicate names

# start training
ac_kwargs = dict(hidden_sizes=[64, 64], activation=torch.nn.ReLU)
logger_kwargs = dict(output_dir='log', exp_name='stander')

ppo(env, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs,
    steps_per_epoch=5000, epochs=250)

# load training data
_, get_action = load_policy_and_env('log')

# start
env_test = gym.make('InvertedPendulum-v2')
run_policy(env_test, get_action)
