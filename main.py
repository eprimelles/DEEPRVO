from environment.Env import DeepNav
from global_DQN.replay_buffer import Replay_Buffer
from global_DQN.DQN import global_DQN

from MADQNTF.replay_buffer import Replay_Buffer as RB_TF
from MADQNTF.MADQN import MADQN_TF
#from MADQN.dqn import DQN_RVO


N_AGENTS = 2
N_EPISODES = 10000
ALGORITHM = 'dqn_tf'
MAX_LENGTH = 10000
BATCH_SIZE = 128
ENV = DeepNav(N_AGENTS, 0)
ALGORITHMS = {
    'globalDQN' : [global_DQN(2, N_AGENTS * 4, 9), Replay_Buffer(N_AGENTS * 4, N_AGENTS, MAX_LENGTH, BATCH_SIZE)],
    'dqn' : 0, # config DQN,
    'dqn_tf' : [MADQN_TF(2, N_AGENTS * 4, 9), RB_TF(N_AGENTS * 4, N_AGENTS, MAX_LENGTH, BATCH_SIZE)]
}

program, rb = ALGORITHMS[ALGORITHM]

program.train(ENV, rb, N_EPISODES)
#print(program.test(ENV))
