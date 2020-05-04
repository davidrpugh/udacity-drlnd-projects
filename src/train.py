import typing

import numpy as np
import torch
from torch import nn, optim
from unityagents import UnityEnvironment

import agents
import experience_replay_buffer
import training


# create the unity environment...
UNITY_ENV = UnityEnvironment(file_name="../environments/Banana.app")
BRAIN_NAME = UNITY_ENV.brain_names[0]
BRAIN = UNITY_ENV.brains[brain_name]

# ...and extract some global state
NUMBER_STATES = BRAIN.vector_observation_space_size
NUMBER_ACTIONS = BRAIN.vector_action_space_size


def beta_annealing_schedule(episode_number: int,
                            initial_beta: float,
                            rate: float):
    """Annealing schedule the strength of correction used with prioritized sampling."""
    return initial_beta + (1 - initial_beta) * (1 - np.exp(-rate * episode_number))


def epsilon_decay_schedule(episode_number: int,
                           decay_factor: float,
                           minimum_epsilon: float) -> float:
    """Decay schedule for the probability that agent chooses an action at random."""
    return max(decay_factor**episode_number, minimum_epsilon)


def make_deep_q_network_fn(number_states: int,
                           number_actions: int,
                           number_hidden_units: int) -> agents.DeepQNetworkFn:
    """Create a function that returns a DeepQNetwork with appropriate input and output shapes."""
    
    def deep_q_network_fn() -> agents.DeepQNetwork:
        deep_q_network = nn.Sequential(
            nn.Linear(in_features=number_states, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=number_actions)
        )
        return deep_q_network
    
    return deep_q_network_fn


def preprocessing_fn(state: agents.UnityState) -> torch.Tensor:
    """Maps external env state repr to agent's internal env state repr."""
    tensor = (torch.Tensor(state)
                   .unsqueeze(dim=0))
    return tensor


_beta_annealing_schedule_kwargs = {
    "rate": 1e-2
}

_replay_buffer_kwargs = {
    "maximum_size": 100000,
    "alpha": 0.5,
    "beta_annealing_schedule": lambda n, b: beta_annealing_schedule(n, b, **_beta_annealing_schedule_kwargs),
    "initial_beta": 0.5,
    "random_state": None,
}
buffer = experience_replay_buffer.PrioritizedExperienceReplayBuffer(**_replay_buffer_kwargs)

_optimizer_kwargs = {
    "lr": 1e-3,
    "betas":(0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}
optimizer_fn = lambda parameters: optim.Adam(parameters, **_optimizer_kwargs)

_deep_q_network_kwargs = {
    "number_states": NUMBER_STATES,
    "number_actions": NUMBER_ACTIONS,
    "number_hidden_units": 64
}
deep_q_network_fn = make_deep_q_network_fn(**_deep_q_network_kwargs)

_epsilon_decay_schedule_kwargs = {
    "decay_factor": 0.99,
    "minimum_epsilon": 1e-2,
}

_agent_kwargs = {
    "number_actions": NUMBER_ACTIONS, 
    "optimizer_fn": optimizer_fn,
    "preprocessing_fn": preprocessing_fn,
    "experience_replay_buffer": buffer,
    "deep_q_network_fn": deep_q_network_fn,
    "epsilon_decay_schedule": lambda n: epsilon_decay_schedule(n, **_epsilon_decay_schedule_kwargs),
    "batch_size": 64,
    "gamma": 0.95,
    "update_frequency": 4,
    "seed": None,
}
double_dqn_agent = agents.DeepQAgent(**_agent_kwargs)
