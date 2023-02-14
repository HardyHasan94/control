"""
Constants
---------
This file contains hyperparameters and constants. For each environment,
its dictionary of parameters must exist here in order to train agents on it.
"""

HYPERPARAMS = {
    'BipedalWalker-v3': {
        'n_steps': 1_000_000,  # env steps
        'evaluation_frequency': 100,  # evaluate agent every that many episodes.
        'actor_lr': 0.0001,
        'critic_lr': 0.0001,
        'gamma': 0.99,
        'tau': 0.001,
        'minibatch': 128,
        'buffer_size': 1_000_000,
        'n_neurons': [400, 300],  # number of neurons for each of the hidden layers
        'learning_starts': 20_000,  # i.e. play randomly for that many steps
        'random_process_parameters': [0, 0.1]  # [mean, stddev]
    },

    'Pendulum-v1': {
        'n_steps': 10_000,
        'evaluation_frequency': 25,
        'actor_lr': 0.001,
        'critic_lr': 0.002,
        'gamma': 0.99,
        'tau': 0.005,
        'minibatch': 64,
        'buffer_size': 10_000,
        'n_neurons': [256, 256],
        'learning_starts': 500,
        'random_process_parameters': [0, 0.2]
    },

    'LunarLanderContinuous-v2': {
        'n_steps': 500_000,
        'evaluation_frequency': 50,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'gamma': 0.99,
        'tau': 0.003,
        'minibatch': 128,
        'buffer_size': 500_000,
        'n_neurons': [400, 300],
        'learning_starts': 10_000,
        'random_process_parameters': [0, 0.1]
    },
    'HalfCheetah-v4': {
        'n_steps': 1_000_000,
        'evaluation_frequency': 100,
        'actor_lr': 0.0001,
        'critic_lr': 0.0001,
        'gamma': 0.99,
        'tau': 0.001,
        'minibatch': 128,
        'buffer_size': 1_000_000,
        'n_neurons': [400, 300],
        'learning_starts': 20_000,
        'random_process_parameters': [0, 0.1]
    }
}

# =============== END OF FILE ===============
