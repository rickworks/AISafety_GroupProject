
'''
based on template by Gili Karni
https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
'''

import time
import numpy as np

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent:
    def __init__(self, env, path=None):
        self.env=env
        self.state_shape=env.observation_space.shape
        self.action_shape=env.action_space.n
        self.gamma=0.99
        self.alpha=1e-4
        self.learning_rate=0.01

        if not path:
            self.model=self._build_model()
        else:
            self.model=self.load_model(path)
            
        self.states, self.gradients, self.rewards, self.probs \
            = [], [], [], []
    ##


    def hot_encode_action(self, action):
        '''encoding the actions into a binary list'''
        action_encoded=np.zeros(self.action_shape, np.float32)
        action_encoded[action]=1
        return action_encoded
    ##
    

    def remember(self, state, action, action_prob, reward):
        '''stores observations'''
        encoded_action=self.hot_encode_action(action)
        self.gradients.append(encoded_action-action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)
    ##


    def _build_model(self):
        ''' builds the model using keras'''
        model=Sequential()

        model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(self.action_shape, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=self.learning_rate))
        
        return model
    ##


    def get_action(self, state):
        '''samples the next action based on the policy probabilty distribution 
        of the actions'''

        state=state.reshape([1, state.shape[0]])
        action_probability_distribution = \
            self.model.predict(state).flatten()
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

        states = np.vstack(self.states)

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

        for episode in range(n_episodes):
            state=self.env.reset()
            done=False
            episode_reward=0
            
            while not done:
                action, prob=self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.remember(state, action, prob, reward)
                state = next_state
                episode_reward += reward
                
                if render and episode > n_episodes - 10:
                    self.env.render()
                    time.sleep(0.02)
                    
                if done:
                    if episode % rollout_n==0:
                        self.update_policy()
                        
            print('=============')
            print(f'episode = {episode}')
            print(f'total reward = {episode_reward}')
                        
            episode_rewards[episode] = episode_reward
                        
        return episode_rewards
    ##


if __name__=='__main__':

    import gym
    env = gym.make('CartPole-v0')

    seed = 0
    np.random.seed(seed); tf.random.set_random_seed(seed), env.seed(seed)

    agent = Agent(env)

    episode_rewards = agent.train(250,render=True)

    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.savefig('cartpole_scores.png')

