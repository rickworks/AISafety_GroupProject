
"""
defined Human, which does automated comparisons between trajectory segments of a pong game.

For use if it turns out to be too time-consuming to train the thing with real human comparisons.

Implements simple rules so won't be as good as an actual human.
"""

import numpy as np

from utils import ball_moving_left, bat_moving_towards_ball

class Human:
    """ does automated comparisons between trajectory segments of a pong game. """
    
    def judge_pong_segments(self,
                            trajectory_segment1,
                            trajectory_segment2):
        """
        given two trajectory_segments (a list of Pong-Atari2600
        observations), judges which is better based on a simple 
        rule defined in self.pong_segment_score()
        """
        try:
            score1 = self.pong_segment_score(trajectory_segment1)
            score2 = self.pong_segment_score(trajectory_segment2)
            if None in [score1, score2]:
                raise
        except:
            return 'n/a'

        if abs(score1 - score2) < 2:
            return '1=2'
        if score1 > score2:
            return '1>2'
        if score1 < score2:
            return '1<2'
    ##


    def pong_segment_score(self,
                           trajectory_segment):
        """
        score = proportion of steps (when ball is moving left) that the bat is moving towards the ball on the y-axis.
        """
        score_history = []
        for step_i in range(len(trajecory_segment) - 1):

            initial_state = trajectory_segment[step_i]
            final_state = trajectory_segment[step_i + 1]

            if ball_moving_left(final_state):
                score_history.append(
                    bat_moving_towards_ball(initial_state, final_state)
                )

        if len(score_history) > 0:
            return (np.sum(score_history)
                    / len(score_history))
    ##
