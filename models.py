import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PolicyNetwork(nn.Module):
    """
    A compact feedforward policy network for environments with discrete actions.

    This network encodes a state vector into a probability distribution over
    discrete actions. It uses a single hidden fully-connected layer with ReLU
    activation, followed by a linear output layer and a softmax to produce
    action probabilities.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Number of discrete actions (output dimension).

    Returns:
        torch.Tensor: A tensor of action probabilities with shape (..., action_dim).

    Example:
        >>> net = PolicyNetwork(state_dim=4, hidden_dim=128, action_dim=2)
        >>> probs = net(torch.randn(1, 4))  # shape (1, 2), sums to 1 along last dim
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

class Qnet(nn.Module):
    """
    A compact feedforward Q-network for environments with discrete actions.

    This network encodes a state vector into Q-values for each discrete action.
    It uses a single hidden fully-connected layer with ReLU activation, followed
    by a linear output layer that produces Q-values.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Number of discrete actions (output dimension).
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

class VAnet(nn.Module):
    """
    A dueling network architecture for value-based reinforcement learning.

    This network separates the estimation of the state value and the advantage for each action.
    The final Q-value for each action is computed by combining the state value and the advantage,
    with the mean advantage subtracted for stability.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Number of discrete actions (output dimension).

    Returns:
        torch.Tensor: A tensor of Q-values for each action.

    Example:
        >>> net = VAnet(state_dim=4, hidden_dim=128, action_dim=2)
        >>> q_values = net(torch.randn(1, 4))  # shape (1, 2)
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        advantage = self.fc_A(x)
        value = self.fc_V(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class ActorCriticNet(nn.Module):
    """
    A simple actor-critic network for environments with discrete actions.

    This network shares a hidden layer between the actor and critic branches.
    The actor branch outputs a probability distribution over actions using a softmax layer,
    while the critic branch outputs a scalar value representing the estimated value of the state.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the shared hidden layer.
        action_dim (int): Number of discrete actions (output dimension).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the action probabilities
        (shape (..., action_dim)) and the state value (shape (..., 1)).

    Example:
        >>> net = ActorCriticNet(state_dim=4, hidden_dim=128, action_dim=2)
        >>> action_probs, state_value = net(torch.randn(1, 4))
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorCriticNet, self).__init__()
        self.shared_layer = nn.Linear(state_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.shared_layer(state))
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        state_value = self.critic_head(features)
        return action_probs, state_value

class DDPGPolicyNet(nn.Module):
    """
    A policy network for Deep Deterministic Policy Gradient (DDPG).

    This network maps a state vector to a continuous action vector. It uses a feedforward
    architecture with one hidden layer and applies a tanh activation function to the output
    to ensure the actions are within the specified bounds.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Dimension of the action space (number of continuous actions).
        action_bound (float): The maximum absolute value for each action.

    Returns:
        torch.Tensor: A tensor of continuous actions with shape (..., action_dim), scaled
        to the range [-action_bound, action_bound].

    Example:
        >>> net = DDPGPolicyNet(state_dim=4, hidden_dim=128, action_dim=2, action_bound=1.0)
        >>> actions = net(torch.randn(1, 4))  # shape (1, 2), values in range [-1.0, 1.0]
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float) -> None:
        super(DDPGPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x)) * self.action_bound
    
class DDPGQnet(nn.Module):
    """
    A Q-network for Deep Deterministic Policy Gradient (DDPG).

    This network estimates the Q-value for a given state-action pair. It uses a feedforward
    architecture with one hidden layer. The input consists of both the state vector and the
    action vector, which are concatenated before being passed through the network.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Dimension of the action space (number of continuous actions).

    Returns:
        torch.Tensor: A tensor of Q-values with shape (..., 1), representing the estimated
        value of the given state-action pair.

    Example:
        >>> net = DDPGQnet(state_dim=4, hidden_dim=128, action_dim=2)
        >>> q_value = net(torch.randn(1, 4), torch.randn(1, 2))  # shape (1, 1)
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(DDPGQnet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
