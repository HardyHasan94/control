"""
Main
----
This file can be called from a command line for either training a
new model or evaluating an existing model.

How to train a new model:
    python3 main.py --train --env_name=BipedalWalker-v3 --model_name=coolWalker --seed=2023

How to evaluate a trained model with rendering:
    python3 main.py --env_name=BipedalWalker-v3 --model_name=coolWalker --n_eval_episodes=1 --render_evaluation
"""

import argparse
import os
from time import perf_counter
from datetime import timedelta

import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import numpy as np

import ddpg
import models
import utils
import constants

# To silence TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser(description="")
parser.add_argument('--train', required=False, action='store_true',
                    help="Whether to train a new model (use argument) or evualte a saved model (omit argument).")
parser.add_argument('--seed', required=False, type=int, default=2023, help="Provide a seed for reproducing results.")
parser.add_argument('--model_name', required=False, type=str, default='',
                    help="A name to save the model to (for training) or a name of a saved model (for evaluation).")
parser.add_argument('--env_name', required=True, type=str, default='Pendulum-v1',
                    help="Full name of the environment.")
parser.add_argument('--n_runs', required=False, type=int, default=1,
                    help="For training multiple agent on the same env with successive seeds.")
parser.add_argument('--n_eval_episodes', required=False, type=int, default=1,
                    help="Number of episodes to evaluate an agent for.")
parser.add_argument('--render_evaluation', required=False, action='store_true', default=False,
                    help="Whether to render the evaluation.")
parser.add_argument('--record_evaluation', required=False, action='store_true', default=False,
                    help="Whether to record the evaluation. Cannot render simultaneously.")
parser.add_argument('--log_data', required=False, action='store_true', default=False,
                    help="Whether to log collected training data to wandb. wandb must be installed and logged in, "
                         "and a project created.")


if __name__ == "__main__":
    args = parser.parse_args()
    training, seed, model_name, env_name, render_evaluation, n_runs, n_eval_episodes, record_evaluation, log_data = \
        args.train, args.seed, args.model_name, args.env_name, args.render_evaluation, args.n_runs, \
        args.n_eval_episodes, args.record_evaluation, args.log_data

    params = constants.HYPERPARAMS[env_name]

    env = gym.make(env_name)
    params['env_name'] = env_name
    params['seed'] = seed
    params['n_actions'] = env.action_space.shape[0]
    params['action_bounds'] = [env.action_space.low, env.action_space.high]
    params['obs_shape'] = (1, *env.observation_space.shape)

    # ----------- Training -----------
    if training:
        params['model_name'] = f"{env_name}_{seed}_{model_name}" if model_name != '' else f"{env_name}_{seed}"
        all_episode_returns = []
        all_episode_lengths = []
        all_critic_losses = []

        for i in range(n_runs):
            utils.set_seed(seed+i)
            agent = models.Agent(params)
            target_agent = models.Agent(params)

            # -- Building models --
            s = np.array([env.observation_space.sample()])
            a, at = agent.actor.call(s), target_agent.actor.call(s)
            q, qt = agent.critic.call([s, a]), target_agent.critic.call([s, at])
            # ---------------------

            start_time = perf_counter()
            episode_returns, episode_lengths, critic_losses = ddpg.train(
                agent=agent, target_agent=target_agent, params=params, model_name=model_name, log_data=log_data)
            end_time = perf_counter()

            print(f"\nRun:{i} Training time: _ _ _ _ _{str(timedelta(seconds=end_time - start_time))}_ _ _ _ _")

            all_episode_lengths.append(episode_lengths)
            all_episode_returns.append(episode_returns)
            all_critic_losses.append(critic_losses)

        if not log_data:
            utils.plot_training(all_episode_returns=all_episode_returns, all_episode_lengths=all_episode_lengths,
                                all_critic_losses=all_critic_losses, figure_name=f'{env_name}_{seed}')

    # ---------- Evaluation ----------
    else:
        agent = models.Agent(params)
        agent.load_model(model_name)
        episode_rewards = ddpg.evaluate(agent=agent,
                                        env_name=env_name,
                                        obs_shape=params['obs_shape'],
                                        n_actions=params['n_actions'],
                                        n_episodes=n_eval_episodes,
                                        render=render_evaluation,
                                        record=record_evaluation)
        print(f"The model {model_name} got a mean return of {np.mean(episode_rewards).round()}!")


# =============== END OF FILE ===============
