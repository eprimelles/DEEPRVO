from .models import Q_Net
import tensorflow as tf
from keras.models import clone_model, load_model
from keras.losses import Huber
from keras.optimizers import Adam
import numpy as np

np.random.seed(7)
class global_DQN:

    def __init__(self, n_agents, state_space, action_space, epsilon=0.8, gamma=0.99, lr=1e-04, path='') -> None:
        
        self.n_agents = n_agents
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma= gamma

        self.model = Q_Net(self.state_space, (self.n_agents, self.action_space))
        self.t_model = clone_model(self.model)
        self.optimizer = Adam(learning_rate=lr)

    def poliy(self, s):

        if self.epsilon > np.random.rand():
            return np.random.randint(0, self.action_space, self.n_agents)
        
        return self.act(s)
        
    def act(self, s):
        return np.squeeze(np.argmax(self.model.predict(s, verbose=0), 2))
    
    def update_target(self):
        self.t_model = clone_model(self.model)

    def save(self):
        self.model.save(f'{self.path}/gloabal_dqn.h5')

    def load(self):
        self.model = load_model(f'{self.path}/gloabal_dqn.h5')

    @tf.function
    def learn(self, s, a, r, s_1, done):
        
        t_q_values = tf.reduce_max(self.t_model(s_1), axis=2)
        target = r +  self.gamma *  t_q_values
        
        mask = tf.one_hot(a, self.action_space)

        with tf.GradientTape() as tape:
            q_values = self.model(s, training=True)
            q_values = tf.reduce_sum(tf.multiply(q_values, mask), axis=2)
            loss = Huber()(q_values, target)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss

    def apply_decay(self, decay):

        self.epsilon -= decay

    def train(self, env, replay_buffer, n_episodes=50000):

        s = env.reset()
        loss = 0
        losses = []
        for i in range(n_episodes):    
            while 1:
                s_e = np.reshape(s[0], (1, len(s[0])))
                a = self.poliy(s_e)
                s_1, rwd, done = env.step(a)
                
                replay_buffer.store(s[0], a, s_1[0], rwd, done)

                s = s_1

                if replay_buffer.isReady():
                    states, actions, rewards, states_1, dones = replay_buffer.sample()
                    loss = self.learn(states, actions, rewards, states_1, dones)
                    
                if done:
                    s = env.reset()
                    self.apply_decay(1/(n_episodes + 100))
                    losses.append(loss)
                    if i % 1000 == 0:
                        self.save()
                        self.update_target()
                        print(f'Episode {i} / {n_episodes}, Last reward: {rwd}, Epsilon: {self.epsilon}, Loss: {loss} ')

                        rr, dd = self.test(env)

                        n = 'not'
                        print(f'Test episode ended with reward {rr}. Ended with {(not dd) * n} succes')

    def test(self, env):
        s = env.reset()
        done = False

        while not done:
            s_e = np.reshape(s[0], (1, len(s_e)))
            a = self.act(s_e)
            s, r, done = env.step(a)

        return r, done



