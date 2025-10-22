import numpy as np
from typing import List, Dict, Tuple
import gymnasium as gym
import math
import swanlab

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils import set_seed, plot_learning_curve
from models import ActorCriticNet
from memory import ReplayBuffer

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    This class implements the PPO algorithm for reinforcement learning. It includes methods
    for action selection, deterministic action selection, and policy updates using the PPO
    objective. The agent uses an Actor-Critic network to estimate both the policy (actor)
    and the value function (critic).

    Attributes:
        actor_critic_net (ActorCriticNet): The shared Actor-Critic network.
        optimizer (torch.optim.Optimizer): Optimizer for updating the network parameters.
        criterion (torch.nn.MSELoss): Loss function for the critic (value function).
        gamma (float): Discount factor for future rewards.
        ppo_epochs (int): Number of epochs for PPO updates.
        ppo_batch (int): Batch size for PPO updates.
        ppo_eps (float): Clipping parameter for PPO updates.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).
        critic_loss_coef (float): Coefficient for the critic loss in the total loss.
        entropy_coef (float): Coefficient for the entropy loss in the total loss.
        device (torch.device): Device to run the computations (CPU or GPU).
        count (int): Counter for the number of updates performed.
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float,
                 gamma: float, gae_lambda: float, critic_loss_coef: float, entropy_coef: float, 
                 ppo_epochs: int, ppo_batch: int, ppo_eps: float, device: torch.device) -> None:
        """
        Initializes the PPOAgent with the given hyperparameters and network architecture.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in the shared layer of the Actor-Critic network.
            action_dim (int): Dimension of the action space (number of discrete actions).
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).
            critic_loss_coef (float): Coefficient for the critic loss in the total loss.
            entropy_coef (float): Coefficient for the entropy loss in the total loss.
            ppo_epochs (int): Number of epochs for PPO updates.
            ppo_batch (int): Batch size for PPO updates.
            ppo_eps (float): Clipping parameter for PPO updates.
            device (torch.device): Device to run the computations (CPU or GPU).
        """
        self.actor_critic_net = ActorCriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.ppo_batch = ppo_batch
        self.ppo_eps = ppo_eps
        self.gae_lambda = gae_lambda
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device
        self.count = 0

    @torch.no_grad()
    def take_action(self, state: np.ndarray) -> tuple[int, float]:
        """
        Selects an action based on the current policy and returns the action and its log probability.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            tuple[int, float]: A tuple containing the selected action and its log probability.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor_critic_net(state_tensor)[0]
        action_dist = Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()
    
    @torch.no_grad()
    def act_deterministic(self, state: np.ndarray) -> int:
        """
        Selects an action deterministically based on the current policy (greedy action).

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The selected action.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor_critic_net(state_tensor)[0]
        action = torch.argmax(probs, dim=1)
        return action.item()
    
    def update(self, transitions: Dict[str, any]) -> float:
        """
        Updates the policy and value function using the PPO objective.

        Args:
            transitions (Dict[str, any]): A dictionary containing the following keys:
                - 'states': Batch of states.
                - 'actions': Batch of actions taken.
                - 'rewards': Batch of rewards received.
                - 'next_states': Batch of next states.
                - 'dones': Batch of done flags (indicating episode termination).
                - 'log_probs': Batch of log probabilities of the actions taken.

        Returns:
            float: The average loss over the PPO update.
        """
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transitions['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], dtype=torch.float).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor(transitions['log_probs'], dtype=torch.float).unsqueeze(1).to(self.device)

        with torch.no_grad():
            old_values = self.actor_critic_net(states)[1]
            td_target = rewards + self.gamma * self.actor_critic_net(next_states)[1] * (1 - dones)
            td_delta = td_target - old_values

            advantages = torch.zeros_like(rewards).to(self.device)
            advantage = 0.0
            for i in reversed(range(len(td_delta))):
                advantage = td_delta[i] + self.gamma * self.gae_lambda * advantage * (1 - dones[i])
                advantages[i] = advantage
            
            returns = advantages + old_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = states.size(0)
        avg_loss = 0.0
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.ppo_batch):
                mb_idx = indices[start:start + self.ppo_batch]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx].squeeze(1)
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                # mb_td_target = td_target[mb_idx]
                mb_returns = returns[mb_idx]

                dist = Categorical(self.actor_critic_net(mb_states)[0]) # new_probs
                new_log_probs = dist.log_prob(mb_actions).unsqueeze(1)
                entropy = dist.entropy().mean()
                state_values = self.actor_critic_net(mb_states)[1]

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_eps, 1 + self.ppo_eps) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = self.criterion(state_values, mb_returns)

                entropy_loss = -entropy
                total_loss = actor_loss + self.critic_loss_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic_net.parameters(), 0.5)
                self.optimizer.step()

                self.count += 1
                avg_loss += total_loss.item()
        avg_loss /= (self.ppo_epochs * math.ceil(T / self.ppo_batch))

        return avg_loss
    
def evaluate(agent: PPOAgent, env: gym.Env, n_episodes: int = 5) -> float:
    """
    Evaluates the performance of the PPO agent in the given environment.

    This function runs the agent in the environment for a specified number of episodes
    using deterministic actions (greedy policy). It calculates the average reward obtained
    across all episodes, which serves as a measure of the agent's performance.

    Args:
        agent (PPOAgent): The PPO agent to be evaluated.
        env (gym.Env): The environment in which the agent will be evaluated.
        n_episodes (int): The number of episodes to run for evaluation. Default is 5.

    Returns:
        float: The average reward obtained across all evaluation episodes.
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        state_np = np.array(state)
        done = False
        while not done:
            action = agent.act_deterministic(state_np)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state_np = np.array(next_state)
    avg_reward = total_reward / n_episodes
    return avg_reward

def train_PPO(agent: PPOAgent, env: gym.Env, config: Dict[str, any],
              replay_buffer: ReplayBuffer) -> List[Dict[str, any]]:
    """
    Trains the PPO agent in the specified environment.

    This function implements the training loop for the PPO algorithm. It collects experiences
    by interacting with the environment, stores them in a replay buffer, and periodically updates
    the policy and value function using the PPO objective. The training process continues until
    the total number of timesteps reaches the specified limit.

    Args:
        agent (PPOAgent): The PPO agent to be trained.
        env (gym.Env): The environment in which the agent will interact and learn.
        config (Dict[str, any]): A dictionary containing training configuration parameters, including:
            - 'lr': Learning rate for the optimizer.
            - 'total_timesteps': Total number of timesteps for training.
            - 'hidden_dim': Number of hidden units in the Actor-Critic network.
            - 'gamma': Discount factor for future rewards.
            - 'gae_lambda': Lambda parameter for Generalized Advantage Estimation (GAE).
            - 'ppo_epochs': Number of epochs for PPO updates.
            - 'ppo_batch': Batch size for PPO updates.
            - 'ppo_eps': Clipping parameter for PPO updates.
            - 'rollout_steps': Number of steps to collect before updating the policy.
            - 'critic_loss_coef': Coefficient for the critic loss in the total loss.
            - 'entropy_coef': Coefficient for the entropy loss in the total loss.
            - 'buffer_size': Maximum capacity of the replay buffer.
            - 'eval_freq': Frequency (in episodes) for evaluation.
            - 'env_name': Name of the environment.
            - 'seed': Random seed for reproducibility.
            - 'device': Device to run the computations (CPU or GPU).
        replay_buffer (ReplayBuffer): A replay buffer to store and sample experiences.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing training statistics, where each dictionary includes:
            - 'steps': The total number of steps taken.
            - 'reward': The total reward obtained in an episode.
            - 'seed': The random seed used for the training run.
        List[Dict[str, any]]: A list of dictionaries containing evaluation statistics, where each dictionary includes:
            - 'steps': The total number of steps taken at the time of evaluation.
            - 'reward': The average reward obtained during evaluation.
            - 'seed': The random seed used for the training run.
    """
    print("--- Start PPO Training ---")
    return_list: List[Dict] = []
    return_list_eval: List[Dict] = []
    total_steps = 0
    i_episode = 0

    while total_steps < config['total_timesteps']:
        episode_return = 0.0
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
            state_np = np.array(next_state)
            episode_return += reward
            total_steps += 1

            if len(replay_buffer) >= config['rollout_steps']:
                b_s, b_a, b_r, b_ns, b_d, b_l = replay_buffer.sample(batch_size=0, sample_all=True)
                transitions_dict = {
                    'states': b_s, 'actions': b_a, 'rewards': b_r,
                    'next_states': b_ns, 'dones': b_d, 'log_probs': b_l
                }
                loss = agent.update(transitions_dict)
                swanlab.log({"Train/Loss": loss}, step=agent.count)

                replay_buffer.clear()

        return_list.append({'steps': total_steps, 'reward': episode_return, 'seed': config['seed']})
        swanlab.log({"Return/by_Episode": episode_return}, step=i_episode)
        swanlab.log({"Return/by_Step": episode_return}, step=total_steps)

        if i_episode % config['eval_freq'] == 0:
            eval_reward = evaluate(agent, env, n_episodes=5)
            swanlab.log({"Eval/Average_Return": eval_reward}, step=total_steps)
            print(f"Episode: {i_episode}, Steps: {total_steps}/{config['total_timesteps']}, "
                  f"Eval Avg Return: {eval_reward:.2f}")
            return_list_eval.append({'steps': total_steps, 'reward': eval_reward, 'seed': config['seed']})
    
    env.close()
    print("--- PPO Training Finished ---")
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
            project="PPO-for-CartPole-v1",
            experiment_name=f"PPO-CartPole-v1-{seed}",
            config={
                "lr": 5e-4,
                "total_timesteps": 40000,
                "hidden_dim": 128,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ppo_epochs": 4,
                "ppo_batch": 64,
                "ppo_eps": 0.2,
                "rollout_steps": 2048,
                "critic_loss_coef": 0.5,
                "entropy_coef": 0.01,
                "buffer_size": 1000000,
                "eval_freq": 20,
                "env_name": 'CartPole-v1',
                "seed": seed,
                "device": device_str
            },
        )
        
        config = swanlab.config

        set_seed(config['seed'])
        replay_buffer = ReplayBuffer(capacity=config['buffer_size'])

        env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = PPOAgent(state_dim, config['hidden_dim'], action_dim, config['lr'],
                    config['gamma'], config['gae_lambda'],  config['critic_loss_coef'],
                    config['entropy_coef'], config['ppo_epochs'], config['ppo_batch'],
                    config['ppo_eps'], device)

        print(f"Using device: {config['device']}")

        single_run_data, single_run_data_eval = train_PPO(agent, env, config, replay_buffer)

        all_seeds_data_list.extend(single_run_data) # Dict{'steps': int, 'reward': float, 'seed': int}
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()
    
    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['PPO'] = all_seeds_data_list
    multi_algo_data_eval['PPO'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data,
                        title="PPO Performance on CartPole-v1",
                        output_filename="ppo_cartpole_performance_shaded.png")

    plot_learning_curve(experiments_data=multi_algo_data_eval,
                        title="PPO Evaluation Performance on CartPole-v1",
                        output_filename="ppo_cartpole_evaluation_performance_shaded.png")

