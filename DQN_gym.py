import numpy as np
from typing import List, Tuple, Dict
import gymnasium as gym
import math
import swanlab

import torch
import torch.optim as optim
import torch.nn as nn

from utils import set_seed, plot_learning_curve
from models import Qnet
from memory import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for environments with discrete actions.

    This agent maintains both an online Q-network and a target Q-network for stable training.
    It supports epsilon-greedy action selection, Q-value estimation, and network updates
    using experience batches. The agent periodically synchronizes the target network with
    the online network.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Number of discrete actions.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        target_update (int): Frequency (in steps) to update the target network.
        device (torch.device): Device to run the networks on (CPU or GPU).
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float,
                 gamma: float, target_update: int, device: torch.device):
        """
        Initializes the DQN agent with Q-networks, optimizer, and training parameters.

        Args:
            state_dim (int): Dimension of the input state vector.
            hidden_dim (int): Number of units in the hidden layer.
            action_dim (int): Number of discrete actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            target_update (int): Frequency to update the target network.
            device (torch.device): Device to run the networks on.
        """
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.target_update = target_update
        self.count = 0
        self.device = device

    @torch.no_grad()
    def take_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Selects an action using the epsilon-greedy strategy.

        Args:
            state (np.ndarray): The current state of the environment.
            epsilon (float): Probability of choosing a random action.

        Returns:
            int: The selected action index.
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            return self.q_net(state_tensor).argmax(dim=1).item()
        
    @torch.no_grad()
    def max_q_value(self, state: np.ndarray) -> int:
        """
        Returns the maximum Q-value for the given state.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The maximum Q-value for the state.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(state_tensor).max(dim=1)[0].item()
    
    def update(self, transition_dict: Dict[str, any]) -> float:
        """
        Updates the Q-network using a batch of transitions.

        Args:
            transition_dict (Dict[str, any]): A dictionary containing batches of states,
                actions, rewards, next_states, and dones.

        Returns:
            float: The computed DQN loss value.
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(dim=1)[0].unsqueeze(1).detach()
        q_target = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = self.criterion(q_values, q_target)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

        return dqn_loss.item()

def evaluate(agent: DQNAgent, env: gym.Env, n_episodes: int) -> float:
    """
    Evaluates the DQN agent over a specified number of episodes.

    Args:
        agent (DQNAgent): The DQN agent to evaluate.
        env (gym.Env): The environment in which to evaluate the agent.
        n_episodes (int): The number of episodes to run for evaluation.

    Returns:
        float: The average reward obtained over the evaluation episodes.
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        state_np = np.array(state)
        done = False
        while not done:
            action = agent.take_action(state_np, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_np = np.array(next_state)
            done = terminated or truncated
            total_reward += reward
            state_np = next_state_np
    avg_reward = total_reward / n_episodes
    return avg_reward 

def train_DQN(agent: DQNAgent, env: gym.Env, config: dict,
              replay_buffer: ReplayBuffer) -> List[Dict]:
    """
    Trains a DQN agent in the specified environment using experience replay.

    This function runs the training loop for the DQN agent, interacting with the environment,
    collecting experiences, updating the agent's Q-network, and logging training metrics.
    Epsilon-greedy exploration is used, and the agent is periodically evaluated.

    Args:
        agent (DQNAgent): The DQN agent to be trained.
        env (gym.Env): The environment in which the agent interacts.
        config (dict): A dictionary containing training hyperparameters such as
            'total_timesteps', 'batch_size', 'epsilon_start', 'epsilon_end',
            'epsilon_decay_steps', and 'eval_freq'.
        replay_buffer (ReplayBuffer): The buffer for storing and sampling experiences.

    Returns:
        List[Dict]: A list of dictionaries containing training statistics for each episode,
        such as total steps and episode rewards.
    """
    print("--- Start DQN Training ---")
    return_list: List[Dict] = []
    return_list_eval: List[Dict] = []
    max_q_value = 0.0
    total_steps = 0
    i_episode = 0

    while total_steps < config['total_timesteps']:
        episode_return = 0.0
        state, _ = env.reset(seed=config['seed'])
        state_np = np.array(state)
        done = False
        i_episode += 1

        while not done:
            epsilon = config['epsilon_end'] + (config['epsilon_start'] - config['epsilon_end']) * \
                                               math.exp(-1. * total_steps / config['epsilon_decay_steps'])
            # fraction = min(1.0, total_steps / config['epsilon_decay_steps'])
            # epsilon = config['epsilon_start'] - fraction * (config['epsilon_start'] - config['epsilon_end'])
            
            action = agent.take_action(state_np, epsilon)

            current_max_q = agent.max_q_value(state_np)
            max_q_value = 0.005 * current_max_q + (1 - 0.005) * max_q_value

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_np = np.array(next_state)

            replay_buffer.push(state_np, action, reward, next_state_np, done, _)
            state_np = next_state_np
            episode_return += reward
            total_steps += 1

            if total_steps > config['learning_starts']:
                b_s, b_a, b_r, b_ns, b_d, _ = replay_buffer.sample(config['batch_size'])
                transition_dict = {
                    'states': b_s, 'actions': b_a,
                    'rewards': b_r, 'next_states': b_ns, 'dones': b_d
                }
                loss = agent.update(transition_dict)
                swanlab.log({"Train/Loss": loss}, step=agent.count)
        
        return_list.append({'steps': total_steps, 'reward': episode_return, 'seed': config['seed']})
        swanlab.log({"Return/by_Episode": episode_return}, step=i_episode)
        swanlab.log({
            "Return/by_Step": episode_return,
            "Train/Epsilon_by_Step": epsilon,
            "Train/Max_Q_Value": max_q_value,
        }, step=total_steps)

        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(agent, env, n_episodes=5)
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"Episode: {i_episode}, Steps: {total_steps}/{config['total_timesteps']}, "
                  f"Eval Avg Return: {eval_reward:.2f}, Epsilon: {epsilon:.3f}")
            return_list_eval.append({'steps': total_steps, 'reward': eval_reward, 'seed': config['seed']})
    
    env.close()
    print("--- DQN Training Finished ---")
    return return_list, return_list_eval

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    seeds = [1, 42, 100, 2024]
    all_seeds_data_list = []
    all_seeds_data_eval_list = []
    multi_algo_data = {}
    multi_algo_data_eval = {}

    for seed in seeds:
        print(f"\n===== 开始运行 Seed: {seed} =====\n")

        run = swanlab.init(
            project="DQN-for-CartPole-v1",
            experiment_name=f"DQN-CartPole-v1-{seed}",
            config={
                "lr": 5e-4,
                "total_timesteps": 40000,
                "hidden_dim": 64,
                "gamma": 0.99,
                "epsilon_start": 0.9,
                "epsilon_end": 0.01,
                "epsilon_decay_steps": 8000,
                "target_update": 500,
                "buffer_size": 10000,
                "batch_size": 64,
                "learning_starts": 1000,
                "eval_freq": 20,
                "env_name": 'CartPole-v1',
                "seed": seed,
                "device": device_str
            },
        )
        
        config = swanlab.config

        set_seed(config['seed'])
        replay_buffer = ReplayBuffer(config['buffer_size'])

        env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DQNAgent(state_dim, config['hidden_dim'], action_dim, config['lr'],
                    config['gamma'], config['target_update'], device)

        print(f"Using device: {config['device']}")

        single_run_data, single_run_data_eval = train_DQN(agent, env, config, replay_buffer)

        all_seeds_data_list.extend(single_run_data) # Dict{'steps': int, 'reward': float, 'seed': int}
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()
    
    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['DQN'] = all_seeds_data_list
    multi_algo_data_eval['DQN'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data,
                        title="DQN Performance on CartPole-v1",
                        output_filename="dqn_cartpole_performance_shaded.png")

    plot_learning_curve(experiments_data=multi_algo_data_eval,
                        title="DQN Evaluation Performance on CartPole-v1",
                        output_filename="dqn_cartpole_evaluation_performance_shaded.png")