import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from dqn_environment import GridEnvironment
from game import Game
import numpy as np
from enums import TileState
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def get_subset(array: np.ndarray, position: np.ndarray, proximity_dimension: np.ndarray, fill=TileState.OBSTACLE.value):
    delta_low, delta_high = position - proximity_dimension, position + proximity_dimension + 1
    output = np.full(proximity_dimension*2 + 1, fill_value=fill)

    # Calculate the valid range of the subset within the array boundaries
    valid_low = np.maximum(0, delta_low)
    valid_high = np.minimum(array.shape, delta_high)

    # Calculate the corresponding valid range in the output array
    output_low = valid_low - delta_low
    output_high = valid_high - delta_low

    # Assign the valid subset from the array to the output array
    output[output_low[0]:output_high[0], output_low[1]:output_high[1]] = array[valid_low[0]:valid_high[0], valid_low[1]:valid_high[1]]
    return output

def create_agent(game: Game):
    env = GridEnvironment(game)

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4


    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    env.reset()
    proximity_dimension = np.array([9, 9])
    n_observations = (proximity_dimension*2+1).prod()

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    max_steps = game.grid.size * 2
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 500

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state = env.reset()
        state = get_subset(state, game.position, proximity_dimension)
        state = state.flatten()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for step in range(max_steps):
            action = select_action(policy_net, env, device, state, eps_end=EPS_END, eps_decay=EPS_DECAY, eps_start=EPS_START)
            observation, reward, terminated = env.step(action.item())
            game.render(lazy=False)
            observation = get_subset(observation, game.position, proximity_dimension)
            observation = observation.flatten()
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(optimizer, policy_net, target_net, device, memory, gamma=GAMMA, batch_size=BATCH_SIZE)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            print(f"episode: {i_episode}, step: {step}")
        if i_episode % 50 and i_episode != 0:
            torch.save(target_net, f"home.{i_episode // 50}.pth")         
    torch.save(target_net, "home.final.pth")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.hidden_size = int(n_observations // 2)
        self.num_layers = 3
        self.gru_layers = nn.ModuleList()
        self.input_layer = nn.GRU(n_observations, self.hidden_size)
        for i in range(self.num_layers - 1):
            self.gru_layers.append(nn.GRU(self.hidden_size, self.hidden_size))
        # self.fc_layers = nn.ModuleList()

        # self.input_layer = nn.Linear(n_observations, self.hidden_size)
        # for _ in range(self.num_layers - 1):
        #     self.fc_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.fc = nn.Linear(self.hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x, h = self.input_layer(x)
        for layer in self.gru_layers:
            x, h = layer(x, h)
        # x = self.input_layer(x)
        # for fc in self.fc_layers:
        #     x = fc(x)
        out = self.fc(x)
        return out

def select_action(policy_net: nn.Module, env: GridEnvironment, device, state, eps_end: float, eps_start: float, eps_decay: float):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * env.steps_done / eps_decay)
    env.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.int8)


episode_durations = [] 

def optimize_model(optimizer: optim.Optimizer, policy_net: nn.Module, target_net: nn.Module, device, memory: ReplayMemory, gamma: float, batch_size: int):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()