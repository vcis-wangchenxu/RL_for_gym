import numpy as np
from typing import Tuple, List, Dict
import gymnasium as gym
import math
import swanlab

import torch
import torch.nn as nn  
import torch.optim as optim
from torch.distributions import Categorical

from utils import set_seed, plot_learning_curve
from models import PolicyNetwork
from memory import ReplayBuffer

class REINFORCEAgent:
    """
    REINFORCE agent for policy gradient reinforcement learning with discrete actions.

    This agent uses a feedforward policy network to parameterize the action distribution.
    It supports stochastic and deterministic action selection, and updates the policy
    using the REINFORCE algorithm with Monte Carlo returns. The agent normalizes returns
    for improved training stability.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Number of discrete actions.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        device (torch.device): Device to run the network on (CPU or GPU).
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float,
                 gamma: float, device: torch.device) -> None:
        """
        Initializes the REINFORCE agent with a policy network, optimizer, and training parameters.

        Args:
            state_dim (int): Dimension of the input state vector.
            hidden_dim (int): Number of units in the hidden layer.
            action_dim (int): Number of discrete actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            device (torch.device): Device to run the network on.
        """
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device
        self.count = 0

    @torch.no_grad()
    def take_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Selects an action stochastically according to the policy's probability distribution.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            Tuple[int, float]: The selected action and its log probability.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        probs = self.policy_net(state_tensor)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()
    
    @torch.no_grad()
    def act_deterministic(self, state: np.ndarray) -> int:
        """
        Selects an action deterministically (the action with the highest probability).

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The selected action index.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        probs = self.policy_net(state_tensor)
        return torch.argmax(probs, dim=-1).item()
    
    def update(self, transition_dict: Dict[str, any]) -> float:
        """
        Updates the policy network using the REINFORCE algorithm and Monte Carlo returns.

        Args:
            transition_dict (Dict[str, any]): A dictionary containing batches of states,
                actions, and rewards collected from an episode.

        Returns:
            float: The computed policy loss value.
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).to(self.device)
        rewards = transition_dict['rewards']
        
        probs = self.policy_net(states)
        action_dist = Categorical(probs)
        log_probs = action_dist.log_prob(actions)

        G = 0
        returns = []
        for r in rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = - (log_probs * returns).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.count += 1

        return policy_loss.item()

def evaluate(agent: REINFORCEAgent, env: gym.Env, n_episodes: int = 5) -> float:
    """
    Evaluates the agent's performance over a specified number of episodes.
    Args:
        agent (REINFORCEAgent): The REINFORCE agent to evaluate.
        env (gym.Env): The environment in which to evaluate the agent.
        n_episodes (int, optional): The number of episodes to run for evaluation. Default is 5.
    
    Returns:
        float: The average reward obtained over the evaluation episodes.
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _  = env.reset()
        state_np = np.array(state)
        done = False
        while not done:
            action = agent.act_deterministic(state_np)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_np = np.array(next_state)
            total_reward += reward
    avg_reward = total_reward / n_episodes
    return avg_reward

def train_REINFORCE(agent: REINFORCEAgent, env: gym.Env, config: dict,
                    replay_buffer: ReplayBuffer) -> List[Dict[str, any]]:
    """
    Trains a REINFORCE agent in the specified environment using Monte Carlo policy gradients.

    This function runs the training loop for the REINFORCE agent, interacts with the environment,
    collects episode trajectories, updates the policy network using the REINFORCE algorithm,
    and logs training metrics. The agent is periodically evaluated, and training statistics
    are logged to SwanLab.

    Args:
        agent (REINFORCEAgent): The REINFORCE agent to be trained.
        env (gym.Env): The environment in which the agent interacts.
        config (dict): A dictionary containing training hyperparameters such as
            'total_timesteps', 'eval_freq', and 'seed'.
        replay_buffer (ReplayBuffer): The buffer for storing episode experiences.

    Returns:
        Tuple[List[Dict], List[Dict]]: Two lists of dictionaries containing training statistics and evaluation results,
        such as total steps, episode rewards, and seed for each episode.
    """
    print("--- Start REINFORCE Training ---")
    return_list: List[Dict] = []
    return_list_eval: List[Dict] = []
    total_steps = 0
    i_episode = 0

    while total_steps < config['total_timesteps']:
        episode_return = 0
        state, _ = env.reset(seed=config['seed'])
        state_np = np.array(state)
        done = False
        i_episode += 1

        while not done:
            action, log_prob = agent.take_action(state_np)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_np = np.array(next_state)

            replay_buffer.push(state_np, action, reward, next_state_np, done, log_prob)
            state_np = next_state_np
            episode_return += reward
            total_steps += 1

        b_s, b_a, b_r, _, _, _ = replay_buffer.sample(batch_size=0, sample_all=True)
        transition_dict = {
            'states': b_s, 'actions': b_a, 'rewards': b_r
        }
        loss = agent.update(transition_dict)
        replay_buffer.clear()

        return_list.append({'steps': total_steps, 'reward': episode_return, 'seed': config['seed']})
        swanlab.log({"Train/Loss": loss}, step=agent.count)
        swanlab.log({"Return/by_Episode": episode_return}, step=i_episode)
        swanlab.log({"Return/by_Step": episode_return}, step=total_steps)

        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(agent, env, n_episodes=5)
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"Episode: {i_episode}, Steps: {total_steps}/{config['total_timesteps']}, "
                  f"Eval Avg Return: {eval_reward:.2f}")
            return_list_eval.append({'steps': total_steps, 'reward': eval_reward, 'seed': config['seed']})
    
    env.close()
    print("--- REINFORCE Training Finished ---\n")
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
            project="REINFORCE-for-CartPole-v1",
            experiment_name=f"REINFORCE-CartPole-v1-{seed}",
            config={
                "lr": 5e-4,
                "total_timesteps": 40000,
                "hidden_dim": 64,
                "gamma": 0.99,
                "buffer_size": 1000000,
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

        agent = REINFORCEAgent(state_dim, config['hidden_dim'], action_dim, config['lr'],
                    config['gamma'], device)

        print(f"Using device: {config['device']}")

        single_run_data, single_run_data_eval = train_REINFORCE(agent, env, config, replay_buffer)

        all_seeds_data_list.extend(single_run_data) # Dict{'steps': int, 'reward': float, 'seed': int}
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()
    
    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['REINFORCE'] = all_seeds_data_list
    multi_algo_data_eval['REINFORCE'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data,
                        title="REINFORCE Performance on CartPole-v1",
                        output_filename="reinforce_cartpole_performance_shaded.png")

    plot_learning_curve(experiments_data=multi_algo_data_eval,
                        title="REINFORCE Evaluation Performance on CartPole-v1",
                        output_filename="reinforce_cartpole_evaluation_performance_shaded.png")