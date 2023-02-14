# Deep Deterministic Policy Gradients

Implementation of the DDPG algorithm using tensorflow.keras and training agents on several environments.

The DDPG algorithm is based on the DQN algorithm and DPG algorithm, where the agent plays in continuous-based-action environments, and the agent model consists of two networks, an Actor network that is used to sample actions, and a Critic network that is used to compute the action-value Q.

## Structure

```
├── README.md
├── constants.py
├── ddpg.py
├── main.py
├── media/
├── memory.py
├── models.py
├── requirements.txt
├── trained_models/
├── utils.py
└── videos/
```


## Usage
For training an agent on OpenAI gym environments, such as `BipedalWalker-v3`, run

```python3 main.py --train --env_name=BipedalWalker-v3 --model_name='coolWalker'  --seed=2023```

The agent models will be saved at `trained_models/BipedalWalker-v3_2023_coolWalker`.

For evaluating that agent run

```python3 main.py --env_name=BipedalWalker-v3 --model_name=BipedalWalker-v3_2023_coolWalker --render_evaluation```

For full list of command-line arguments and their descriptions run ```python3 main.py -h```.

For training agents on MuJoCo environments, it needs to be installed manually first. See `https://github.com/deepmind/mujoco`.


## Results
Agents three different environments are trained. These are Lunar Lander, Bipedal Walker and Half Cheetah. Below are the training episodic returns. For the first two envs three agents are trained with different seeds and the graphs shows average return and the standard error is the shaded area of the runs. For HalfCheetah only one agent is trained.

![alt text](media/lunarlander.pdf?raw=true "LunarLander")
![alt text](media/bipedalwalker.pdf?raw=true "BipedalWalker")
![alt text](media/half_cheetah.pdf?raw=true "HalfCheetah")

## Author

- Hardy Hasan


## Resources
Explanation of policy gradient RL and algorithms can be found at:
- https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

