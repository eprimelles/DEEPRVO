import tensorflow as tf
import numpy as np


from .DQN import DQN_Agent


class MADQN_TF:

    def __init__(self, n_agents, state_space, action_space, epsilon=0.8, gamma=0.99, lr=1e-04, path='') -> None:
        
        self.n_agents = n_agents
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.path = path

        self.agents = [DQN_Agent(i, self.n_agents, self.state_space, self.action_space, self.epsilon, self.epsilon,
                                 self.lr, self.path)
                                 for i in range(self.n_agents)]

    def choose_actions(self, s):
        return [i.act(s[:, i.id]).numpy() for i in self.agents] 
    
    def apply_decay(self, decay):
        for i in self.agents:
            i.apply_decay(decay)

        self.epsilon -= decay

    def update_networks(self):
        for i in self.agents:
            i.update_target()

    def policy(self, s):
        return [agnt.policy(s[:, id]) for id, agnt in enumerate(self.agents)] 

    def save(self):
        for i in self.agents:
            i.save()

    def load(self):
        for i in self.agents:
            i.load()      
    
    def train(self, env, replay_buffer, n_episodes):

        
        s = env.reset()
        rwds = []
        loss = [0] * self.n_agents
        losses = []
        for i in range(n_episodes):    
            while 1:
                s_e = np.reshape(s, (1, len(s), len(s[0])))
                a = self.policy(s_e)
                
                aa = a.copy()
                s_1, rwd, success,  done = env.step(a)
                
                rwds.append(rwd)
                
                replay_buffer.store(s, aa, s_1, rwd, done)

                s = s_1
                
                if replay_buffer.isReady():
                    
                    states, actions, rewards, states_1, dones = replay_buffer.sample()
                    for id, agnt in enumerate(self.agents):
                        loss[id] = float(agnt.learn(states[:, id], actions[:, id], rewards[:, id], states_1[:, id], dones))
                    
                if done:
                    s = env.reset()
                    self.apply_decay(1/(n_episodes + 100))
                    losses.append(loss)
                    print(f'Episode {i} / {n_episodes}, Success: {success} Last reward: {np.mean(rwds, axis=0)}, Epsilon: {self.epsilon}, Loss: {loss} ')
                    rwds = []
                    if i % 1000 == 0:
                        self.save()
                        self.update_networks()
                        #print(f'Episode {i} / {n_episodes}, Last reward: {rwd}, Epsilon: {self.epsilon}, Loss: {loss} ')

                        #rr, dd = self.test(env)

                        #n = 'not'
                        #print(f'Test episode ended with reward {rr}. Ended with {(not dd) * n} succes')
                    break

    def test(self, env):

        self.load()
        print('...Testing Episode... ')
        s = env.reset()

        done = False
        rwds = []
        while not done:

            s_e = np.reshape(s, (1, len(s), len(s[0])))
            a = self.choose_actions(s_e)
            
            s, r, success, done = env.step(a)
            rwds.append(r)

        print(f'Episode ended with Success: {success}, Mean RWD: {np.mean(rwds, axis=0)}, Time: {env.time}, Baseline: {env.baseline}')