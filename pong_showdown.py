
"""
Script to run Pong-Atari2600 between two agents.

Also has a couple of baseline agents for illustration, a random agent and an OpenAI baseline agent.

<player1/2> must define class Agent with attribute .showdown_step():
   showdown_step(state = Pong-Atari2600 observation,
                 reward = scalar float)
   returns action = Pong-Atari2600 action, which is a
                    binary list of length 8, e.g.
                    [0,1,0,0,1,0,1,1]
"""

import time
from copy import deepcopy

import numpy as np
import retro

from openai_baseline import load_baseline

#import <player1>
#import <player2>

def main():
    """ does thing """

    #player1 = <player1>.Agent()
    #player2 = <player2>.Agent()

    player1 = TrainedBaselineAgent()
    player2 = RandomBaselineAgent()

    # setting up enviroment
    env = retro.make(game='Pong-Atari2600', players=2)
    observation = env.reset()
    done = False
    reward_player1, reward_player2 = (0, 0)

    # game loop
    while not done:

        action_player1 = player1.showdown_step(observation,
                                               reward_player1)
        action_player2 = player2.showdown_step(mirror_observation(observation),
                                               reward_player2)

        observation, reward, done, info = env.step(
            np.concatenate((action_player1, action_player2))
        )

        reward_player1, reward_player2 = tuple(reward)

        env.render()
        time.sleep(0.015)

        if done:
            obs = env.reset()
            env.close()
            print('Final score: '+str(info))
##


def mirror_observation(observation):
    """
    flip observation on x-axis and swap colour of bats,
    to make player2 feel like they're player1.
    """
    observation = deepcopy(observation)

    # flipping x-axis (the enviroment is not quite symmetric
    # so leave last column alone)
    x_extent = observation.shape[1]
    x_reflect_almost = lambda x: (x if x == x_extent-1 else
                                  x_extent-1 - x)

    obs_copy = deepcopy(observation)
    for x in range(x_extent):
        observation[:, x, :] = obs_copy[:, x_reflect_almost(x), :]

    # swap colours of bats
    left_bat_colour = [92, 186, 92]
    right_bat_colour = [213, 130, 74]

    left_bat_pixel_coords = np.where(
        (observation == left_bat_colour).all(axis=2))
    right_bat_pixel_coords = np.where(
        (observation == right_bat_colour).all(axis=2))

    observation[left_bat_pixel_coords] = right_bat_colour
    observation[right_bat_pixel_coords] = left_bat_colour

    return observation
##


class RandomBaselineAgent:
    """ an agent to play pong that always just does random actions """

    def showdown_step(self,
                      observation,
                      reward):
        """
        args:
        observation = matrix of pixels from pong (ignored)
        reward = float (ignored)
        returns:
        action = binary array of length 8
        """
        return np.random.randint(0, 2, 8)
##


class TrainedBaselineAgent:
    """ here's one I made earlier """

    def __init__(self):
        self.model = load_baseline()
    ##

    def showdown_step(self,
                      observation,
                      reward):
        """
        args:
        observation = matrix of pixels from pong
        reward = float (ignored)
        returns:
        action = binary array of length 8
        """
        actions, values, observation, _ = self.model.step(observation)
        return actions[0]
    ##


if __name__ == "__main__":
    main()
