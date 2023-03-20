import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import torch.optim as optim
import math
from itertools import count
# from ReplayMemory import ReplayMemory
import utils
# Transition = namedtuple('Transition',
                        # ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4



class DQN_RVO(nn.Module):
    def __init__(self, state_space, action_space) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, action_space))

    def forward(self, x):
        x = self.network(x)
        return x

class MADQN(object):
    def __init__(self, n_agents, state_space, action_space, epsilon=0.8, gamma=0.99, lr=1e-04, path='') -> None:
        self.n_agents = n_agents
        self.state_space = state_space
        self.steps_done = 0 
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma= gamma  
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.policy_net = DQN_RVO(state_space=state_space, action_space=(n_agents, action_space)).to(self.device)
        self.target_net = DQN_RVO(state_space=state_space, action_space=(n_agents, action_space)).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)     

    def select_action(self, state, env, policy_net):        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with T.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1).cpu().numpy()
        else:
            return T.tensor([[random.sample(sorted(env.p_actions),1)]], device=self.device, dtype=T.long).cpu().numpy()
        
    
    def optimize_model(self, replayBuffer, optimizer, policy_net, target_net):
        if len(replayBuffer) < BATCH_SIZE:
            return
        transitions = replayBuffer.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=T.bool)
        non_final_next_states = T.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = T.zeros(BATCH_SIZE, device=self.device)
        with T.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # state value or 0 in case the state was final.
        next_state_values = T.zeros(BATCH_SIZE, device=self.device)
        with T.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        T.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    
    def train(self, env, replay_buffer, n_episodes=50000):
        state = env.reset()
        loss = 0
        losses = []
        self.steps_done = 0
        episode_durations = []        

        for i_episode in range(n_episodes):
            # Initialize the environment and get it's state
            state = env.reset()
            state = T.tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state, env, self.policy_net)                
                observation, reward, terminated  = env.step(action)
                reward = T.tensor([reward], device=self.device)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = T.tensor(observation, dtype=T.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                replay_buffer.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(replay_buffer, self.optimizer, self.policy_net, self.target_net)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)            
                    break

            print('Complete')