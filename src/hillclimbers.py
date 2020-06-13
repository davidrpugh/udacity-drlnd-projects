import copy
import typing

import numpy as np
import torch
from torch import distributions, nn


UnityState = np.ndarray
Policy = nn.Module
PolicyFn = typing.Callable[[], Policy]
PreprocessingFn = typing.Callable[[UnityState], torch.Tensor] 
Action = int


class UnityAgent:
    
    def __init__(self,
                 policy_fn: PolicyFn,
                 gamma: float = 1.0,
                 preprocessing_fn: typing.Optional[PreprocessingFn] = None,
                 seed: typing.Optional[int] = None):
        if torch.cuda.is_available():
            self._device = torch.device("cuda") 
        else:
            self._device = torch.device("cpu")
        self._gamma = gamma
        self._policy = policy_fn()
        _ = self._policy.to(self._device)
        
        # converts np.ndarray with shape (n_states,) to torch.Tensor with shape (1, n_states)
        if preprocessing_fn is None:
            self._preprocessing_fn = lambda state: torch.Tensor(state).unsqueeze(dim=0) 
        else:
            self._preprocessing_fn = preprocessing_fn
            
        if seed is None:
            self._random_state = np.random.RandomState()
        else:
            self._random_state = np.random.RandomState(seed)
            torch.manual_seed(seed)
            
            # if using CUDA need to set cuDNN flags
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
    def __call__(self, state: UnityState) -> Action:
        """Apply the current policy function to the state and returns the action."""
        state_tensor = (self._preprocessing_fn(state)
                            .to(self._device))
        action = (self._policy(state_tensor)
                      .cpu())
        return action

    def evaluate(self, rewards) -> None:
        """Evaluate rewards generated from following the current policy while interacting with an environment."""
        raise NotImplementedError
        
    def load_state_dict(self, state_dict: typing.Dict[str, torch.Tensor]) -> None:
        """Update the parameters of the policy function using the state_dict."""
        copied_state_dict = copy.deepcopy(state_dict) # avoid sharing mutable state_dict!
        _ = self._policy.load_state_dict(copied_state_dict)


class HillClimber(UnityAgent):
    
    def __init__(self,
                 policy_fn: PolicyFn,
                 gamma: float = 1.0,
                 preprocessing_fn: typing.Optional[PreprocessingFn] = None,
                 seed: typing.Optional[int] = None,
                 sigma: float = 1.0):
        super().__init__(policy_fn, gamma, preprocessing_fn, seed)
        self._max_discounted_reward = -float("inf")
        copied_state_dict = copy.deepcopy(self._policy.state_dict()) # avoid storing mutable state!
        self._best_policy_state_dict = copied_state_dict 
        self._sigma = sigma
        
    def __ge__(self, other):
        return not self.__lt__(other)
    
    def __lt__(self, other):
        return self._max_discounted_reward < other._max_discounted_reward

    @property
    def best_policy_state_dict(self):
        """The state_dict parameterizing the best policy function found thus far."""
        return self._best_policy_state_dict
    
    @property
    def score(self):
        """The maximum discounted reward achieved thus far."""
        return self._max_discounted_reward
    
    def evaluate(self, rewards):
        """Evaluate rewards from following current policy when interacting with an environment."""
        discounted_reward = sum(self._gamma**i * reward for i, reward in enumerate(rewards))
        if discounted_reward >= self._max_discounted_reward:
            self._max_discounted_reward = discounted_reward
            copied_state_dict = copy.deepcopy(self._policy.state_dict()) # avoid storing mutable state!
            self._best_policy_state_dict = copied_state_dict           
        else:
            _ = self.load_state_dict(self._best_policy_state_dict)
        with torch.no_grad():
            for parameter in self._policy.parameters():
                parameter.add_(torch.randn_like(parameter), alpha=self._sigma)
