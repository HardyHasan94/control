"""
Main
----
This file contains the main function that can be called from
a command line for either training a new model or evaluating an
existing model on some environment.
"""

import argparse

import ddpg

parser = argparse.ArgumentParser(description="")
parser.add_argument('--seed', required=True, type=int, default=2022, help="")
parser.add_argument('--problem', required=True, type=str, default='FACES', choices=['FACES', 'SINE', 'MNIST'],
                    help="")
parser.add_argument('--wandb', required=False, action='store_true', default=False,
                    help="")


def main():
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    problem_name, seed, log_wandb = args.problem, args.seed, args.wandb


# =============== END OF FILE ===============
