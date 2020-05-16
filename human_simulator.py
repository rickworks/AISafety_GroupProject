
"""

"""

import numpy as np

def parse_pong_state(state):
    """ 
    translates a pong state (matrix of rgb values for each pixel) to more natural information (coordinate of player's bats and ball)
    """
    # cutting off top and bottom panels
    state = state[34:194,:,:]

    # removing colour
    monochrome_state = np.sum(state,axis=2)

    #print(np.unique(monochrome_state))
    background_shade = 233
    left_bat_shade = 417
    right_bat_shade = 370
    ball_shade = 708

    coord_from_shade = lambda shade: np.mean(
        np.argwhere(monochrome_state==shade),
        axis=0 
    ) + np.array([34,0])

    return { 'left_bat_height' : \
             coord_from_shade( left_bat_shade ),
             'right_bat_height' : \
             coord_from_shade( right_bat_shade ),
             'ball_coord' : \
             coord_from_shade( ball_shade )
    }
##


def bat_moving_towards_ball(initial_state,
                            final_state):
    """
    returns true if left bat moves towards the ball between initial and final state
    """
    initial_state_info = parse_pong_state(initial_state)
    final_state_info = parse_pong_state(final_state)
    
    ball_height = ( initial_state_info['ball_coord'][0] +
                    final_state_info['ball_coord'][0] )/2
    
    bat_ball_heightdiff_initial = abs(
        initial_state_info['left_bat_height'] - ball_height )
    bat_ball_heightdiff_final = abs(
        final_state_info['left_bat_height'] - ball_height )

    return bat_ball_heightdiff_final < bat_ball_heightdiff_initial
##


def ball_moving_left(initial_state,
                     final_state):
    """ 
    returns true if the ball is further left in final_state
    than initial state
    """
    initial_state_info = parse_pong_state(initial_state)
    final_state_info = parse_pong_state(final_state)

    initial_ball_x = initial_state_info['ball_coord'][1]
    final_ball_x = final_state_info['ball_coord'][1]

    return final_ball_x < initial_ball_x
##


class Human:
    def judge_pong_segments(self,
                            trajectory_segment1,
                            trajectory_segment2):
        """ 
        given two trajectory_segments (a list of Pong-Atari2600
        observations), judges which is better based on a simple rule defined in self.pong_segment_score()
        """
        try:
            score1 = self.pong_segment_score(trajectory_segment1)
            score2 = self.pong_segment_score(trajectory_segment2)
            if None in [score1,score2]:
                raise
        except:
            return 'n/a'
        
        if abs( score1 - score2 ) < 2:
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
        for step_i in range(len(trajecory_segment)-1):
            
            initial_state = trajectory_segment[step_i]
            final_state = trajectory_segment[step_i+1]

            if ball_moving_left(final_state):
                score_history.append(
                    bat_moving_towards_ball(initial_state,final_state)
                )

        if len(score_history) > 0:
            return ( np.sum( score_history )
                     / len( score_history ) )
        else:
            return None
    ##
