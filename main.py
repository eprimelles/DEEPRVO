from environment.Env import DeepNav
from global_DQN.replay_buffer import Replay_Buffer
from global_DQN.DQN import global_DQN
#from MADQN.dqn import DQN_RVO


N_AGENTS = 2
N_EPISODES = 50000
ALGORITHM = 'globalDQN'
MAX_LENGTH = 10000
BATCH_SIZE = 128
ENV = DeepNav(N_AGENTS, 0)
ALGORITHMS = {
    'globalDQN' : [global_DQN(2, N_AGENTS * 4, 9), Replay_Buffer(N_AGENTS * 4, N_AGENTS, MAX_LENGTH, BATCH_SIZE)],
    'dqn' : 0 # config DQN
}

program, rb = ALGORITHMS[ALGORITHM]

program.train(ENV, rb, N_EPISODES)