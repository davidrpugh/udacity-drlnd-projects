import typing

import numpy as np
import torch
from torch import nn, optim

from experience_replay_buffer import Experience, PrioritizedExperienceReplayBuffer


# custom types for public API
UnityState = np.ndarray
Action = int
Reward = float
Done = bool

# custom types used by private Agent API
DeepQNetwork = nn.Module
DeepQNetworkFn = typing.Callable[[], DeepQNetwork]
QValues = torch.Tensor
TDErrors = torch.Tensor


def synchronize_q_networks(q1: DeepQNetwork, q2: DeepQNetwork) -> None:
    """In place, synchronization of q1 and q2."""
    _ = q1.load_state_dict(q2.state_dict())


def select_greedy_actions(states: torch.Tensor, q_network: DeepQNetwork) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    _, actions = q_network(states).max(dim=1, keepdim=True)
    return actions


def evaluate_selected_actions(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: DeepQNetwork) -> torch.Tensor:
    """Compute the Q-values by evaluating the actions given the current states and Q-network."""
    next_q_values = q_network(states).gather(dim=1, index=actions)        
    q_values = rewards + (gamma * next_q_values * (1 - dones))
    return q_values


def double_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q1: DeepQNetwork,
                             q2: DeepQNetwork) -> QValues:
    """Double Q-Learning uses q1 to select actions and q2 to evaluate the selected actions."""
    actions = select_greedy_actions(states, q1)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q2)
    return q_values


def double_q_learning_error(states: torch.Tensor,
                            actions: torch.Tensor,
                            rewards: torch.Tensor,
                            next_states: torch.Tensor,
                            dones: torch.Tensor,
                            gamma: float,
                            q1: DeepQNetwork,
                            q2: DeepQNetwork) -> TDErrors:
    expected_q_values = double_q_learning_update(next_states, rewards, dones, gamma, q1, q2)
    q_values = q1(states).gather(dim=1, index=actions)
    delta = expected_q_values - q_values
    return delta


class UnityAgent:
    
    def __call__(self, state: UnityState) -> Action:
        """Rule for choosing an action given the current state of the environment."""
        raise NotImplementedError

    def save(self, filepath) -> None:
        """Save any important agent state to a file."""
        raise NotImplementedError
        
    def step(self,
             state: UnityState,
             action: Action,
             reward: float,
             next_state: UnityState,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplmentedError


class DeepQAgent(UnityAgent):

    def __init__(self,
                 number_actions: int,
                 deep_q_network_fn: DeepQNetworkFn, 
                 optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                 preprocessing_fn: typing.Callable[[UnityState], torch.Tensor],
                 batch_size: int,
                 experience_replay_buffer: PrioritizedExperienceReplayBuffer,
                 epsilon_decay_schedule: typing.Callable[[int], float],
                 gamma: float,
                 update_frequency: int,
                 seed: int = None) -> None:
        """
        Initialize a DeepQAgent.
        
        Parameters:
        -----------
        state_size (int): the size of the state space.
        action_size (int): the size of the action space.
        number_hidden_units (int): number of units in the hidden layers.
        optimizer_fn (callable): function that takes Q-network parameters and returns an optimizer.
        batch_size (int): number of experience tuples in each mini-batch.
        buffer_size (int): maximum number of experience tuples stored in the replay buffer.
        alpha (float): Strength of prioritized sampling; alpha >= 0.0.
        beta_annealing_schedule (callable): function that takes episode number and returns beta >= 0.
        epsilon_decay_schdule (callable): function that takes episode number and returns 0 <= epsilon < 1.
        alpha (float): rate at which the target q-network parameters are updated.
        gamma (float): Controls how much that agent discounts future rewards (0 < gamma <= 1).
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        seed (int): random seed
        
        """
        self._number_actions = number_actions
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set seeds for reproducibility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self._batch_size = batch_size
        self._memory = experience_replay_buffer
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._gamma = gamma
        
        # initialize Q-Networks
        self._preprocessing_fn = preprocessing_fn
        self._update_frequency = update_frequency
        self._online_q_network = deep_q_network_fn()
        self._target_q_network = deep_q_network_fn()
        synchronize_q_networks(self._target_q_network, self._online_q_network)        
        self._online_q_network.to(self._device)
        self._target_q_network.to(self._device)
        
        # initialize the optimizer
        self._optimizer = optimizer_fn(self._online_q_network.parameters())

        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0
        
    def __call__(self, state: UnityState) -> Action:
        """Agent uses epsilon-greedy policy for choosing actions given the state."""
        preprocessed_state = (self._preprocessing_fn(state)
                                  .to(self._device))
        if not self.has_sufficient_experience():
            action = self._uniform_random_policy(preprocessed_state)
        else:
            epsilon = self._epsilon_decay_schedule(self._number_episodes)
            action = self._epsilon_greedy_policy(preprocessed_state, epsilon)
        return action
           
    def _uniform_random_policy(self, state: torch.Tensor) -> Action:
        """Choose an action uniformly at random."""
        return self._random_state.randint(self._number_actions)
        
    def _greedy_policy(self, state: torch.Tensor) -> Action:
        """Choose an action that maximizes the action_values given the current state."""
        actions = select_greedy_actions(state, self._online_q_network)
        action = (actions.cpu()  # actions might reside on the GPU!
                         .item())
        return action
    
    def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float) -> Action:
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        if self._random_state.random() < epsilon:
            action = self._uniform_random_policy(state)
        else:
            action = self._greedy_policy(state)
        return action
    
    def _ddqn_algorithm(self,
                        idxs: np.ndarray,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_states: torch.Tensor,
                        dones: torch.Tensor,
                        sampling_weights: torch.Tensor) -> None:
        """Double deep Q-network (DDQN) algorithm with prioritized experience replay."""

        # compute the temporal difference errors
        deltas = double_q_learning_error(states,
                                         actions,
                                         rewards,
                                         next_states,
                                         dones,
                                         self._gamma,
                                         self._online_q_network,
                                         self._target_q_network)
        
        # update experience priorities
        priorities = (deltas.abs()
                            .cpu()
                            .detach()
                            .numpy()
                            .flatten())
        self._memory.update_priorities(idxs, priorities + 1e-6) # priorities must be positive!
        
        # compute the mean squared loss
        loss = torch.mean((deltas * sampling_weights)**2)

        # updates the parameters of the online network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        synchronize_q_networks(self._target_q_network, self._online_q_network)
    
    def has_sufficient_experience(self) -> bool:
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        return len(self._memory) >= self._batch_size
    
    def save(self, filepath: str) -> None:
        """
        Saves the state of the DeepQAgent.
        
        Parameters:
        -----------
        filepath (str): filepath where the serialized state should be saved.
        
        Notes:
        ------
        The method uses `torch.save` to serialize the state of the q-network, 
        the optimizer, as well as the dictionary of agent hyperparameters.
        
        """
        checkpoint = {
            "q-network-state": self._online_q_network.state_dict(),
            "optimizer-state": self._optimizer.state_dict(),
            "agent-hyperparameters": {
                "batch_size": self._memory.batch_size,
                "gamma": self._gamma,
                "update_frequency": self._update_frequency
            }
        }
        torch.save(checkpoint, filepath)
        
    def step(self,
             state: UnityState,
             action: Action,
             reward: Reward,
             next_state: UnityState,
             done: Done) -> None:
        """Update internal state after observing effect of action on the environment."""
        experience = Experience(state, action, reward, next_state, done)
        self._memory.add(experience) 

        if done:
            self._number_episodes += 1
        else:
            self._number_timesteps += 1
            
            # every so often the agent should learn from experiences
            if self._number_timesteps % self._update_frequency == 0 and self.has_sufficient_experience():
                idxs, _experiences, _sampling_weights = self._memory.sample(self._batch_size, self._number_episodes)
                
                # unpack the experiences
                _states, _actions, _rewards, _next_states, _dones = tuple(zip(*_experiences))

                # need to preprocess _states/_next_states
                state_tensors = [self._preprocessing_fn(s) for s in _states]
                next_state_tensors = [self._preprocessing_fn(s) for s in _next_states]

                states = (torch.cat(state_tensors, dim=0)
                               .to(self._device))
                actions = (torch.Tensor(_actions)
                                .long()
                                .unsqueeze(dim=1)
                                .to(self._device))
                rewards = (torch.Tensor(_rewards)
                                .unsqueeze(dim=1)
                                .to(self._device))
                next_states = (torch.cat(next_state_tensors, dim=0)
                                    .to(self._device))
                dones = (torch.Tensor(_dones)
                               .unsqueeze(dim=1)
                               .to(self._device))
                
                # reshape sampling weights
                sampling_weights = (torch.Tensor(_sampling_weights)
                                         .view((-1, 1))
                                         .to(self._device))
                
                self._ddqn_algorithm(idxs, states, actions, rewards, next_states, dones, sampling_weights)
