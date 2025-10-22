import numpy as np
from typing import List, Dict
import gymnasium as gym
import math
import swanlab

import torch
import torch.optim as optim
import torch.nn as nn

from utils import set_seed, plot_learning_curve
from models import Qnet
from memory import ReplayBuffer

class DoubleDQNAgent:
    """
    Double Deep Q-Network (Double DQN) agent for environments with discrete actions.

    This agent maintains both an online Q-network and a target Q-network for stable training.
    It uses the Double DQN algorithm to reduce overestimation of Q-values by decoupling
    action selection and evaluation. The agent supports epsilon-greedy action selection,
    Q-value estimation, and network updates using experience batches. The target network
    is periodically synchronized with the online network.

    Args:
        state_dim (int): Dimension of the input state vector.
        hidden_dim (int): Number of units in the hidden layer.
        action_dim (int): Number of discrete actions.
        learnint_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        target_update (int): Frequency (in steps) to update the target network.
        device (torch.device): Device to run the networks on (CPU or GPU).
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learnint_rate: float,
                 gamma: float, target_update: int, device: torch.device) -> None:
        """
        Initializes the Double DQN agent with Q-networks, optimizer, and training parameters.

        Args:
            state_dim (int): Dimension of the input state vector.
            hidden_dim (int): Number of units in the hidden layer.
            action_dim (int): Number of discrete actions.
            learnint_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            target_update (int): Frequency to update the target network.
            device (torch.device): Device to run the networks on.
        """
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learnint_rate)
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
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            return self.q_net(state).argmax(dim=1).item()
    
    @torch.no_grad()
    def max_q_value(self, state: np.array) -> int:
        """
        Returns the maximum Q-value for the given state.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The maximum Q-value for the state.
        """
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(state).max(dim=1)[0].item()
    
    def update(self, transition_dict: Dict[str, np.ndarray]) -> float:
        """
        Updates the Q-network using a batch of transitions with the Double DQN algorithm.

        Args:
            transition_dict (Dict[str, np.ndarray]): A dictionary containing batches of states,
                actions, rewards, next_states, and dones.

        Returns:
            float: The computed Double DQN loss value.
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].unsqueeze(1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action).detach()

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        double_dqn_loss = self.criterion(q_values, q_targets)
        
        self.optimizer.zero_grad()
        double_dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        return double_dqn_loss.item()
    
def dis_to_con(discrete_action: int, env: gym.Env, action_dim: int) -> float:
    """
    Converts a discrete action index to a continuous action value for environments with continuous action spaces.

    This function maps a discrete action (represented by its index) to a corresponding continuous value
    within the environment's action space bounds using linear interpolation.

    Args:
        discrete_action (int): The index of the discrete action to be converted.
        env (gym.Env): The environment instance, used to obtain action space bounds.
        action_dim (int): The total number of discrete actions.

    Returns:
        float: The corresponding continuous action value.
    """
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def evaluate(agent: DoubleDQNAgent, env: gym.Env, n_episodes: int) -> float:
    """
    Evaluates the Double DQN agent's performance over a specified number of episodes.

    This function runs the agent in the environment for a given number of episodes,
    using greedy action selection (epsilon=0), and computes the average total reward
    obtained across all episodes.

    Args:
        agent (DoubleDQNAgent): The Double DQN agent to be evaluated.
        env (gym.Env): The environment in which to evaluate the agent.
        n_episodes (int): The number of episodes to run for evaluation.

    Returns:
        float: The average total reward obtained over the evaluation episodes.
    """
    total_rewards = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        state_np = np.array(state)
        done = False
        while not done:
            action = agent.take_action(state_np, epsilon=0.0)
            action_continuous = dis_to_con(action, env, agent.action_dim)
            next_state, reward, terminated, truncated, _ = env.step([action_continuous])
            next_state_np = np.array(next_state)
            done = terminated or truncated
            total_rewards += reward
            state_np = next_state_np
    return total_rewards / n_episodes

def train_DoubleDQN(agent: DoubleDQNAgent, env: gym.Env, config: dict,
                    replay_buffer: ReplayBuffer) -> List[Dict[str, any]]:
    """
    Trains the Double DQN agent using the specified environment and configuration.

    This function samples experiences from the replay buffer and performs
    optimization steps on the agent's neural network to minimize the
    Double DQN loss.

    Args:
        agent (DoubleDQNAgent): The Double DQN agent to be trained.
        env (gym.Env): The environment in which the agent operates.
        config (dict): Configuration parameters for training.
        replay_buffer (ReplayBuffer): The replay buffer containing past experiences.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing training metrics
        and information for each training iteration.
    """
    print("--- Start DoubleDQN Training ---")
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
            
            action = agent.take_action(state_np, epsilon)
            action_continuous = dis_to_con(action, env, agent.action_dim)

            current_max_q = agent.max_q_value(state_np)
            max_q_value = 0.005 * current_max_q + (1 - 0.005) * max_q_value

            next_state, reward, terminated, truncated, _ = env.step([action_continuous])
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
    print("--- DoubleDQN Training Finished ---")
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
            project="DoubleDQN-for-Pendulum-v1",
            experiment_name=f"DoubleDQN-Pendulum-v1-{seed}",
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
                "env_name": 'Pendulum-v1',
                "seed": seed,
                "device": device_str
            },
        )
        
        config = swanlab.config

        set_seed(config['seed'])
        replay_buffer = ReplayBuffer(config['buffer_size'])

        env = gym.make(config['env_name'], render_mode='rgb_array', max_episode_steps=200)
        state_dim = env.observation_space.shape[0]
        action_dim = 11

        agent = DoubleDQNAgent(state_dim, config['hidden_dim'], action_dim, config['lr'],
                    config['gamma'], config['target_update'], device)

        print(f"Using device: {config['device']}")

        single_run_data, single_run_data_eval = train_DoubleDQN(agent, env, config, replay_buffer)

        all_seeds_data_list.extend(single_run_data) # Dict{'steps': int, 'reward': float, 'seed': int}
        all_seeds_data_eval_list.extend(single_run_data_eval)
        
        swanlab.finish()
    
    print("Starting to plot learning curve across all seeds...")
    multi_algo_data['DoubleDQN'] = all_seeds_data_list
    multi_algo_data_eval['DoubleDQN'] = all_seeds_data_eval_list

    plot_learning_curve(experiments_data=multi_algo_data,
                        title="DoubleDQN Performance on Pendulum-v1",
                        output_filename="double_dqn_pendulum_performance_shaded.png")

    plot_learning_curve(experiments_data=multi_algo_data_eval,
                        title="DoubleDQN Evaluation Performance on Pendulum-v1",
                        output_filename="double_dqn_pendulum_evaluation_performance_shaded.png")
