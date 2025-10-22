from collections import deque
import numpy as np
from typing import Tuple

class ReplayBuffer:
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.

    The replay buffer stores tuples of (state, action, reward, next_state, done, log_prob)
    and allows sampling of random batches for training. It is implemented using a deque
    for efficient appending and clearing of experiences.

    Attributes:
        buffer (deque): A deque to store experience tuples.
    """
    def __init__(self, capacity: int=1e7) -> None:
        """
        Initializes an empty replay buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, 
                   action: int, 
                   reward: float, 
                   next_state: np.ndarray, 
                   done: bool,
                   log_prob: float) -> None:
        """
        Adds a new experience tuple to the replay buffer.

        Args:
            state (np.ndarray): The current state of the environment.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The next state of the environment after the action.
            done (bool): A flag indicating whether the episode has ended.
            log_prob (float): The log probability of the action taken.
        """
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def sample(self, batch_size: int, sample_all: bool = False, sequential: bool = False) -> Tuple[np.ndarray, int, float, np.ndarray, bool, float]:
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample (ignored if sample_all is True).
            sample_all (bool): If True, return all experiences in the buffer.
            sequential (bool): If True, sample experiences in order; otherwise, sample randomly.

        Returns:
            Tuple[np.ndarray, int, float, np.ndarray, bool, float]: A tuple containing
            batches of states, actions, rewards, next_states, dones, and log_probs.
        """
        buffer_length = len(self.buffer)
        if sample_all:
            indices = range(buffer_length)
        elif sequential:
            if batch_size > buffer_length:
                raise ValueError("batch_size exceeds buffer length for sequential sampling.")
            indices = range(batch_size)
        else:
            if batch_size > buffer_length:
                raise ValueError("batch_size exceeds buffer length for random sampling.")
            indices = np.random.choice(buffer_length, batch_size, replace=False)
        states, actions, rewards, next_states, dones, log_probs = zip(*(self.buffer[idx] for idx in indices))
        return (np.array(states),
                actions,
                rewards,
                np.array(next_states),
                dones,
                log_probs)
    
    def clear(self) -> None:
        """
        Clears all experiences from the replay buffer.
        """
        self.buffer.clear()

    def __len__(self) -> int:
        """
        Returns the number of experiences currently stored in the replay buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.buffer)
    
