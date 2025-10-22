import numpy as np
from typing import List, Tuple, Dict
import gymnasium as gym
import math
import swanlab

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils import set_seed, plot_learning_curve
from models import DDPGPolicyNet, DDPGQnet
from memory import ReplayBuffer

class DDPGAgent:
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float, 
                 actor_learning_rate: float, critic_learning_rate: float, gamma: float, 
                 tau: float, device: torch.device):
        self.actor = DDPGPolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor = DDPGPolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

        self.critic = DDPGQnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = DDPGQnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.count = 0

    @torch.no_grad()
    def take_action(self, state: np.ndarray, exploration_noise: float) -> np.ndarray:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state_tensor).cpu().numpy()[0]
        noise = np.random.normal(0, exploration_noise, size=action.shape)
        return (action + noise).clip(-self.actor.action_bound, self.actor.action_bound)
    
    @torch.no_grad()
    def act_deterministic(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state_tensor).cpu().numpy()[0]
        return action
    
    def soft_update(self, net: nn.Module, target_net: nn.Module):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def update(self, transition_dict: Dict[str, any]) -> Tuple[float, float]:
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        critic_loss = self.criterion(self.critic(states, actions), q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        self.count += 1

        return actor_loss.item(), critic_loss.item()

def evaluate(agent: DDPGAgent, env: gym.Env, n_episodes: int = 5) -> float:
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

def train_DDPG(agent: DDPGAgent, env: gym.Env, config: Dict[str, any], 
               replay_buffer: ReplayBuffer) -> List[Dict[str, any]]:
    print("--- Start DDPG Training ---")
    return_list: List[Dict] = []
    return_list_eval: List[Dict] = []
    total_steps = 0
    i_episode = 0

    while total_steps < config['total_timesteps']:
        episode_return  = 0.0
        state, _ = env.reset(seed=config['seed'])
        state_np = np.array(state)
        done = False
        i_episode += 1

        while not done: 
            action = agent.take_action(state_np, config['exploration_noise'])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_np = np.array(next_state)

            replay_buffer.push(state_np, action, reward, next_state_np, done, _)
            state_np = next_state_np
            episode_return += reward
            total_steps += 1

            if len(replay_buffer) >= config['learning_starts']:
                b_s, b_a, b_r, b_ns, b_d, _ = replay_buffer.sample(config['batch_size'])
                transition_dict = {
                    'states': b_s, 'actions': b_a, 'rewards': b_r,
                    'next_states': b_ns, 'dones': b_d
                }
                actor_loss, critic_loss = agent.update(transition_dict)
                swanlab.log({
                    "Train/Actor_Loss": actor_loss,
                    "Train/Critic_Loss": critic_loss
                }, step=agent.count)

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
    print("--- DDPG Training Finished ---")
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
            project="DDPG-for-Pendulum-v1",
            experiment_name=f"DDPG-Pendulum-v1-{seed}",
            config={
                "actor_lr": 3e-4,
                "critic_lr": 3e-3,
                "total_timesteps": 40000,
                "hidden_dim": 128,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": 10000,
                "batch_size": 64,
                "learning_starts": 1000,
                "exploration_noise": 0.1,
                "eval_freq": 20,
                "env_name": 'Pendulum-v1',
                "seed": seed,
                "device": device_str
            },
        )
        
        config = swanlab.config

        set_seed(config['seed'])
        replay_buffer = ReplayBuffer(capacity=config['buffer_size'])

        env = gym.make(config['env_name'], render_mode='rgb_array')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        agent = DDPGAgent(state_dim, config['hidden_dim'], action_dim, action_bound,
                          config['actor_lr'], config['critic_lr'], config['gamma'],
                          config['tau'], device)

        print(f"Using device: {config['device']}")

        single_run_data, single_run_data_eval = train_DDPG(agent, env, config, replay_buffer)

        all_seeds_data_list.extend(single_run_data) # Dict{'steps': int, 'reward': float, 'seed': int}
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()
    
    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['DDPG'] = all_seeds_data_list
    multi_algo_data_eval['DDPG'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data,
                        title="DDPG Performance on Pendulum-v1",
                        output_filename="ddpg_pendulum_performance_shaded.png")

    plot_learning_curve(experiments_data=multi_algo_data_eval,
                        title="DDPG Evaluation Performance on Pendulum-v1",
                        output_filename="ddpg_pendulum_evaluation_performance_shaded.png")

