import numpy as np


class Replay_Buffer:

    def __init__(self, state_space, n_agents, max_length, batch_size) -> None:
        self.state_buffer = np.zeros((max_length, state_space), dtype=np.float32)
        self.action_buffer = np.zeros((max_length, n_agents), dtype=np.int32)
        self.state_1_buffer = np.zeros((max_length, state_space), dtype=np.float32)
        self.reward_buffer = np.zeros((max_length, n_agents), dtype=np.float32)
        self.done_buffer = np.zeros((max_length, 1), dtype=np.float32)
        self.max_lenght = max_length
        self.batch_size = batch_size
        self.indx = 0
        self.ready = False    

    def store(self, s, a, s_1, r, done):

        i = self.indx % self.max_lenght

        self.state_buffer[i] = np.array(s)
        self.state_1_buffer[i] = s_1
        self.action_buffer[i] = a
        
        self.reward_buffer[i] = np.array(r) 
        self.done_buffer[i] = done

        self.indx += 1
        self.ready = self.indx > self.batch_size
    
    def sample (self):
        try:
            assert self.ready
        except:
            print('Buffer not ready')
        indxs = np.random.choice(min(self.indx, self.max_lenght), self.batch_size)
        s = self.state_buffer[indxs]
        a = self.action_buffer[indxs]
        s_1 = self.state_1_buffer[indxs]
        r = self.reward_buffer[indxs]
        dones = self.done_buffer[indxs]

        return s, a, r, s_1, dones
    
    def isReady(self):
        return self.ready
    

