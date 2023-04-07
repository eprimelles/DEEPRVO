from .models import Q_Net
import tensorflow as tf
from keras.models import clone_model, load_model
from keras.losses import Huber
from keras.optimizers import Adam
import numpy as np

np.random.seed(7)
class DQN_Agent:

    def __init__(self, id,  n_agents, state_space, action_space, epsilon=0.8, gamma=0.99, lr=1e-04, path='') -> None:
        
        self.id = id
        self.n_agents = n_agents
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma= gamma
        self.path = path
        self.model = Q_Net(self.state_space, self.action_space)
        self.t_model = clone_model(self.model)
        self.optimizer = Adam(learning_rate=lr)

    def policy(self, s):

        opt = np.random.uniform(0, 1, 1)[0]
        
        if self.epsilon > opt:
            
            return np.random.randint(0, self.action_space)
        
        
        return int(np.array(self.act(s)))
        
    @tf.function
    def act(self, s):
        return tf.squeeze(tf.argmax(self.model(s, training=False), 1))
    
    
    def update_target(self):
        self.t_model = clone_model(self.model)

    def save(self):
        self.model.save(f'{self.path}tf_dqn_{self.id}.h5')

    def load(self):
        self.model = load_model(f'{self.path}gloabal_dqn.h5')

    @tf.function
    def learn(self, s, a, r, s_1, done):
        
        t_q_values = tf.reduce_max(self.t_model(s_1), axis=1)
        target = r +  self.gamma *  t_q_values
        
        mask = tf.one_hot(a, self.action_space)

        with tf.GradientTape() as tape:
            q_values = self.model(s, training=True)
            q_values = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = Huber()(q_values, target)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss

    def apply_decay(self, decay):

        self.epsilon -= decay

    

