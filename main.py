from environment.Env import DeepNav
# from global_DQN.replay_buffer import Replay_Buffer
# from global_DQN.DQN import global_DQN
from MADQN.dqn import MADQN
from MADQN.ReplayMemory import ReplayMemory


N_AGENTS = 2
N_EPISODES = 50000
# ALGORITHM = 'globalDQN'
ALGORITHM = 'dqn'
MAX_LENGTH = 10000
BATCH_SIZE = 128
ENV = DeepNav(N_AGENTS, 0)
ALGORITHMS = {
    'globalDQN' : 0, # [global_DQN(2, N_AGENTS * 4, 9), Replay_Buffer(N_AGENTS * 4, N_AGENTS, MAX_LENGTH, BATCH_SIZE)],
    'dqn' : [MADQN(N_AGENTS, N_AGENTS * 4, 9), ReplayMemory(10000)] # config DQN
}

program, rb = ALGORITHMS[ALGORITHM]

program.train(ENV, rb, N_EPISODES)
