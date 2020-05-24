
'''
based on template by Gili Karni
https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
'''

import sys
sys.path.insert(0,'../AISafety_GroupProject/')
from utils import parse_pong_state

import time
import numpy as np

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.optimizers import Adam

class Agent:
    def __init__(self, env, path=None):
        self.env=env
        #self.state_shape=env.observation_space.shape
        self.preprocessed_state_shape=(160,160,1)
        self.action_shape=env.action_space.n
        #self.gamma=0.99
        self.gamma=0.8
        #self.alpha=1e-4
        self.alpha=1e-30
        #self.learning_rate=0.001
        self.learning_rate=1e-30

        if not path:
            self.model=self._build_model()
        else:
            self.model=self.load_model(path)
            
        self.states, self.gradients, self.rewards, self.probs \
            = [], [], [], []
    ##

    
    def _preprocess_state(self, state):
        return np.mean(state,axis=2)[34:194,:,np.newaxis]
    ##


    def hot_encode_action(self, action):
        '''encoding the actions into a binary list'''
        action_encoded=np.zeros(self.action_shape, np.float32)
        action_encoded[action]=1
        return action_encoded
    ##
    

    def _remember(self, state, action, action_prob, reward):
        '''stores observations'''
        encoded_action=self.hot_encode_action(action)
        self.gradients.append(encoded_action-action_prob)
        preprocessed_state = self._preprocess_state(state)
        self.states.append(preprocessed_state)
        self.rewards.append(reward)
        print(f'distr = {action_prob}')
        self.probs.append(action_prob)
    ##


    def _build_model(self):
        ''' builds the model using keras'''
        model=Sequential()

        model.add(MaxPooling2D(pool_size=(4, 4),
                               strides=(4, 4),
                               input_shape=self.preprocessed_state_shape))
        model.add(Conv2D(24, kernel_size=(4, 4), strides=(2, 2),
                         activation='relu'))
        model.add(Flatten())
        
        model.add(Dense(8, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(self.action_shape, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=self.learning_rate))
        
        return model
    ##


    def get_action(self, state):
        '''samples the next action based on the policy probabilty distribution 
        of the actions'''

        state = self._preprocess_state(state)
        #state = np.full_like(state,0)
        
        action_probability_distribution = \
            self.model.predict([[state]]).flatten()
        action_probability_distribution /= \
            np.sum(action_probability_distribution)

        action=np.random.choice(self.action_shape,
                                1,
                                p=action_probability_distribution)[0]
        
        return action, action_probability_distribution
    ##
    

    def get_discounted_rewards(self, rewards):
        '''calculate q-value'''

        discounted_rewards=[]
        cumulative_total_return=0

        for reward in rewards[::-1]:
            cumulative_total_return=(cumulative_total_return*self.gamma)+reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards=np.mean(discounted_rewards)
        std_rewards=np.std(discounted_rewards)
        norm_discounted_rewards=(discounted_rewards-
                                 mean_rewards)/(std_rewards+1e-7)

        return norm_discounted_rewards
    ##
    

    def update_policy(self):
        ''' Updates parameters in policy network '''
        states = np.array(self.states)

        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        labels = self.probs + \
                 self.alpha * gradients

        self.model.train_on_batch(states, labels)
        
        self.states, self.probs, self.gradients, self.rewards \
            = [], [], [], []
    ##


    def train(self, n_episodes, rollout_n=1, render=False):

        episode_rewards=np.zeros(n_episodes)

        for episode_index in range(n_episodes):
            state=self.env.reset()
            done=False
            episode_reward=0
            
            while not done:
                action, prob=self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)

                self._remember(state, action, prob, reward)
                state = next_state
                episode_reward += reward
                
                if render and episode_index > int(n_episodes/2):
                    self.env.render()
                    time.sleep(0.02)
                    
                if done:
                    if episode_index % rollout_n==0 and episode_index > 0:
                        self.update_policy()
                        
            print('=============')
            print(f'episode # = {episode_index}')
            print(f'total reward = {episode_reward}')
                        
            episode_rewards[episode_index] = episode_reward
                        
        return episode_rewards
    ##


if __name__=='__main__':

    import gym
    env = gym.make('Pong-v0')

    #import retro
    #env = retro.make('Pong-Atari2600',players=2)

    seed = 0
    np.random.seed(seed); tf.random.set_random_seed(seed), env.seed(seed)

    agent = Agent(env)

    episode_rewards = agent.train(20,render=True)

    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.savefig('pong_scores.png')

