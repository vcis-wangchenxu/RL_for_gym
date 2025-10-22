import numpy as np
from typing import List, Dict, Tuple
import gymnasium as gym
import math
import swanlab

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim

from utils import set_seed, plot_learning_curve
from models import ActorCriticNet
from memory import ReplayBuffer

class ActorCriticAgent:
    """
    Actor-Critic agent for reinforcement learning with discrete actions.

    This agent uses a shared neural network to estimate both the policy (actor) and the state value (critic).
    The actor outputs a probability distribution over actions, while the critic estimates the value of the current state.
    The agent supports both stochastic and deterministic action selection, and updates its parameters using
    the actor-critic algorithm with n-step returns and entropy regularization for improved exploration.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the shared hidden layer.
        action_dim (int): Number of discrete actions.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        critic_loss_coef (float): Coefficient for the critic loss term.
        entropy_coef (float): Coefficient for the entropy regularization term.
        device (torch.device): Device to run the network on (CPU or GPU).
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float, 
                 gamma: float, critic_loss_coef: float, entropy_coef: float, device: torch.device) -> None:
        """
        Initializes the Actor-Critic agent with a shared network, optimizer, and training parameters.

        Args:
            state_dim (int): Dimension of the input state vector.
            hidden_dim (int): Number of units in the shared hidden layer.
            action_dim (int): Number of discrete actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            critic_loss_coef (float): Coefficient for the critic loss term.
            entropy_coef (float): Coefficient for the entropy regularization term.
            device (torch.device): Device to run the network on.
        """
        self.actor_critic_net = ActorCriticNet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.actor_critic_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.count = 0
        self.device = device

    @torch.no_grad()
    def take_action(self, state: np.ndarray) -> int:
        """
        Selects an action stochastically according to the policy's probability distribution.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The selected action index.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor_critic_net(state_tensor)[0]
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    @torch.no_grad()
    def act_deterministic(self, state: np.ndarray) -> int:
        """
        Selects an action deterministically (the action with the highest probability).

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The selected action index.
        """
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor_critic_net(state_tensor)[0]
        action = torch.argmax(probs, dim=1)
        return action.item()
        
    def update(self, transition_dict: Dict[str, any]) -> float:
        """
        Updates the actor-critic network using n-step returns and entropy regularization.

        Args:
            transition_dict (Dict[str, any]): A dictionary containing batches of states,
                actions, rewards, next_states, and done flags collected from an episode.

        Returns:
            float: The computed total loss value for the update step.
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).to(self.device)
        rewards = transition_dict['rewards']
        
        next_state_final = torch.tensor(np.array([transition_dict['next_states'][-1]]), dtype=torch.float).to(self.device)
        done_final = transition_dict['dones'][-1]

        with torch.no_grad():
            final_value = self.actor_critic_net(next_state_final)[1] if not done_final else \
                torch.tensor([0.0]).to(self.device)
        
        G = final_value.item()
        returns = []

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        probs, values = self.actor_critic_net(states)
        critic_loss = self.criterion(values.squeeze(dim=1), returns)

        advantages = (returns - values.squeeze(dim=1)).detach()

        action_dist = Categorical(probs)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        actor_loss = - (log_probs * advantages).mean()

        total_loss = actor_loss + self.critic_loss_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.count += 1
        return total_loss.item()
    
def evaluate(agent: ActorCriticAgent, env: gym.Env, n_episodes: int) -> float:
    """
    Evaluates the performance of the Actor-Critic agent over a specified number of episodes.

    This function runs the agent in the environment using deterministic (greedy) action selection,
    accumulates the total reward for each episode, and returns the average reward across all evaluation episodes.

    Args:
        agent (ActorCriticAgent): The agent to be evaluated.
        env (gym.Env): The environment in which the agent is evaluated.
        n_episodes (int): Number of evaluation episodes to run.

    Returns:
        float: The average total reward obtained over all evaluation episodes.
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

def train_ActorCritic(agent: ActorCriticAgent, env: gym.Env, config: Dict[str, any], 
                      replay_buffer: ReplayBuffer) -> List[Dict[str, any]]:
    """
    Trains an Actor-Critic agent in the specified environment using n-step returns and experience replay.

    This function runs the training loop for the Actor-Critic agent, interacts with the environment,
    collects episode trajectories, updates the shared actor-critic network, and logs training metrics.
    The agent uses n-step returns for updates, and training/evaluation statistics are periodically logged to SwanLab.

    Args:
        agent (ActorCriticAgent): The Actor-Critic agent to be trained.
        env (gym.Env): The environment in which the agent interacts.
        config (Dict[str, any]): A dictionary containing training hyperparameters such as
            'total_timesteps', 'n_steps', 'eval_freq', and 'seed'.
        replay_buffer (ReplayBuffer): The buffer for storing episode experiences.

    Returns:
        Tuple[List[Dict], List[Dict]]: Two lists of dictionaries containing training statistics and evaluation results,
        such as total steps, episode rewards, and seed for each episode.
    """
    print("--- Start ActorCritic Training ---")
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
            action = agent.take_action(state_np)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_np = np.array(next_state)

            replay_buffer.push(state_np, action, reward, next_state_np, done, _)
            state_np = next_state_np
            episode_return += reward
            total_steps += 1

            if len(replay_buffer) == config['n_steps'] or done:
                b_s, b_a, b_r, b_ns, b_d, _ = replay_buffer.sample(batch_size=0, sample_all=True)
                transition_dict = {
                    'states': b_s, 'actions': b_a,
                    'rewards': b_r, 'next_states': b_ns, 'dones': b_d
                }
                loss = agent.update(transition_dict)
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
    print("--- ActorCritic Training Finished ---")
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
            project="ActorCritic-for-CartPole-v1",
            experiment_name=f"ActorCritic-CartPole-v1-{seed}",
            config={
                "lr": 5e-4,
                "total_timesteps": 40000,
                "hidden_dim": 128,
                "gamma": 0.99,
                "critic_loss_coef": 0.5,
                "entropy_coef": 0.01,
                "n_steps": 5,
                "eval_freq": 20,
                "env_name": 'CartPole-v1',
                "seed": seed,
                "device": device_str
            },
        )
        
        config = swanlab.config

        set_seed(config['seed'])
        replay_buffer = ReplayBuffer(capacity=config['n_steps'])

        env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = ActorCriticAgent(state_dim, config['hidden_dim'], action_dim, config['lr'],
                    config['gamma'], config['critic_loss_coef'], config['entropy_coef'], device)

        print(f"Using device: {config['device']}")

        single_run_data, single_run_data_eval = train_ActorCritic(agent, env, config, replay_buffer)

        all_seeds_data_list.extend(single_run_data) # Dict{'steps': int, 'reward': float, 'seed': int}
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()
    
    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['ActorCritic'] = all_seeds_data_list
    multi_algo_data_eval['ActorCritic'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data,
                        title="ActorCritic Performance on CartPole-v1",
                        output_filename="actorcritic_cartpole_performance_shaded.png")

    plot_learning_curve(experiments_data=multi_algo_data_eval,
                        title="ActorCritic Evaluation Performance on CartPole-v1",
                        output_filename="actorcritic_cartpole_evaluation_performance_shaded.png")