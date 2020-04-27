import torch
from torch import nn


def synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module) -> None:
    """In place, synchronization of q_network_1 and q_network_2."""
    _ = q_network_1.load_state_dict(q_network_2.state_dict())


def select_greedy_actions(states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    _, actions = q_network(states).max(dim=1, keepdim=True)
    return actions


def evaluate_selected_actions(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: nn.Module) -> torch.Tensor:
    """Compute the Q-values by evaluating the actions given the current states and Q-network."""
    next_q_values = q_network(states).gather(dim=1, index=actions)        
    q_values = rewards + (gamma * next_q_values * (1 - dones))
    return q_values


def double_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q_network_1: nn.Module,
                             q_network_2: nn.Module) -> torch.Tensor:
    """Double Q-Learning uses Q-network 1 to select actions and Q-network 2 to evaluate the selected actions."""
    actions = select_greedy_actions(states, q_network_1)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network_2)
    return q_values


def double_q_learning_error(states: torch.Tensor,
                            actions: torch.Tensor,
                            rewards: torch.Tensor,
                            next_states: torch.Tensor,
                            dones: torch.Tensor,
                            gamma: float,
                            q_network_1: nn.Module,
                            q_network_2: nn.Module) -> torch.Tensor:
    expected_q_values = double_q_learning_update(next_states, rewards, dones, gamma, q_network_1, q_network_2)
    q_values = q_network_1(states).gather(dim=1, index=actions)
    delta = expected_q_values - q_values
    return delta