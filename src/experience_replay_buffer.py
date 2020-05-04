import collections
import typing

import numpy as np


BetaAnnealingSchedule = typing.Optional[typing.Callable[[int, float], float]]
RandomState = typing.Optional[np.random.RandomState]


def sampling_probabilities(priorities: np.ndarray, alpha: float):
    """Sampling probability is increasing function of priority"""
    return priorities**alpha / np.sum(priorities**alpha)
    

def sampling_weights(probabilities: np.ndarray, beta: float, normalize: bool):
    n = probabilities.size
    weights = (n * probabilities)**-beta
    if normalize:
        weights = weights / weights.max()
    return weights


_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)


class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store priority, Experience tuples."""

    def __init__(self,
                 maximum_size: int,
                 alpha: float = 0.0,
                 beta_annealing_schedule: BetaAnnealingSchedule = None,
                 initial_beta: float = 0.0,
                 random_state: RandomState = None) -> None:
        """
        Initialize a PrioritizedExperienceReplayBuffer object.

        Parameters:
        -----------
        maximum_size (int): maximum size of buffer
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        beta_annealing_schedule (BetaAnnealingSchedule): function that takes an episode number and 
            an initial value for beta and returns the current value of beta.
        random_state (np.random.RandomState): random number generator.
        
        """
        self._maximum_size = maximum_size
        self._current_size = 0 # current number of prioritized experience tuples in buffer
        _dtype = [("priority", np.float32), ("experience", Experience)]
        self._buffer = np.empty(self._maximum_size, _dtype)
        self._alpha = alpha
        self._initial_beta = initial_beta
        
        if beta_annealing_schedule is None:
            self._beta_annealing_schedule = lambda n: self._initial_beta
        else:
            self._beta_annealing_schedule = lambda n: beta_annealing_schedule(n, self._initial_beta)

        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._current_size

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha
    
    @property
    def maximum_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._maximum_size
    
    @property
    def alpha(self):
        """Initial strength for sampling correction."""
        return self._initial_beta

    def add(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self._buffer[self._current_size] = (priority, experience)
            self._current_size += 1

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._current_size == 0
    
    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._current_size == self._maximum_size
    
    def sample(self, batch_size: int, episode_number: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._current_size]["priority"]
        sampling_probs = sampling_probabilities(ps, self._alpha)
        
        # use sampling probabilities to compute sampling weights
        beta = self._beta_annealing_schedule(episode_number)
        weights = sampling_weights(sampling_probs, beta, normalize=True)
        
        # randomly sample indicies corresponding to priority, experience tuples
        idxs = np.arange(sampling_probs.size)
        random_idxs = self._random_state.choice(idxs,
                                                size=batch_size,
                                                replace=True,
                                                p=sampling_probs)
        
        # select the experiences and sampling weights
        sampled_experiences = self._buffer["experience"][random_idxs]
        sampled_weights = weights[random_idxs]
        
        return random_idxs, sampled_experiences, sampled_weights

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities
