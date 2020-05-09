
"""
Script to run Pong-Atari2600 between two agents. 

Also has a couple of baseline agents for illustration, a random agent and an OpenAI baseline agent.
"""

import retro
import time
import numpy as np

#import <player1_model>
#import <player2_model>

def main():

    #player1 = player1_model.Model()
    #player2 = player2_model.Model()

    player1 = RandomBaselineModel()
    player2 = RandomBaselineModel()

    # setting up enviroment
    env = retro.make(game='Pong-Atari2600', players=2)
    state = env.reset()
    done= False
    reward_player1, reward_player2 = ( 0, 0 )

    # game loop
    while not done:

        action_player1 = player1.step(state,
                                      reward_player1)
        action_player2 = player2.step(state,
                                      reward_player2)

        state, reward, done, info = env.step(
            np.concatenate((action_player1,action_player2))
        )
        reward_player1, reward_player2 = tuple(reward)
        
        env.render()
        time.sleep(0.015)
        
        if done:
            obs = env.reset()
            env.close()
            print('Final score: '+str(info))
##

            
class RandomBaselineModel:
    """ a model to play pong that always just does random actions """
    
    def step(self,
             state,
             reward):
        """
        args: 
        state = matrix of pixels from pong
        reward = float
        returns:
        action = binary array of length 8
        """
        return np.random.randint(0,2,8)
##

            
            
if __name__ == "__main__":
    main()
