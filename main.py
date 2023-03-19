from environment.Env import DeepNav
# from global_DQN.replay_buffer import Replay_Buffer
# from global_DQN.DQN import global_DQN
from MADQN import dqn


if __name__ == '__main__':

     
    env = DeepNav(2, 0)
    # s = env.reset()
    # rb = Replay_Buffer(8, 2, 10000, 128)

    # while not rb.isReady():

    #     a = (2, 1)

    #     s_1, r, done = env.step(a)

    #     rb.store(s[0], s_1[0], a, r, done)

    # print(rb.sample())
    # s, a, r, s_1, done = rb.sample()

    # model = global_DQN(2, 8, 9)
    # model.learn(s, a, r, s_1, done)
    madqn = dqn.MADQN()
    madqn.train(env=env)
