
"""
some utils
"""

from copy import deepcopy
import numpy as np


### simple observation parsing ###


def parse_pong_state(state):
    """
    translates a pong state (matrix of rgb values for each pixel)
    to more natural information (coordinate of player's bats and ball)
    """
    # cutting off top and bottom panels
    state = state[34:194, :, :]

    # removing colour
    monochrome_state = np.sum(state, axis=2)

    #print(np.unique(monochrome_state))
    background_shade = 233
    left_bat_shade = 417
    right_bat_shade = 370
    ball_shade = 708

    coord_from_shade = lambda shade: np.mean(
        np.argwhere(monochrome_state == shade),
        axis=0
    ) + np.array([34, 0])

    return {'left_bat_height' : \
            coord_from_shade(left_bat_shade)[0],
            'right_bat_height' : \
            coord_from_shade(right_bat_shade)[0],
            'ball_coord' : \
            coord_from_shade(ball_shade)}
##


def bat_moving_towards_ball(initial_state,
                            final_state):
    """
    returns true if left bat moves towards the ball between initial and final state
    """
    initial_state_info = parse_pong_state(initial_state)
    final_state_info = parse_pong_state(final_state)

    ball_height = (initial_state_info['ball_coord'][0] +
                   final_state_info['ball_coord'][0])/2

    bat_ball_heightdiff_initial = abs(
        initial_state_info['left_bat_height'] - ball_height)
    bat_ball_heightdiff_final = abs(
        final_state_info['left_bat_height'] - ball_height)

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


### mirroring observation ###


def mirror_observation(observation):
    """
    flip observation on x-axis and swap colour of bats,
    to make player2 feel like they're player1.
    """
    observation = deepcopy(observation)
    observation = _flip_x_axis(observation)
    observation = _swap_bat_colours(observation)
    
    return observation
##


def _flip_x_axis(observation):
    """ flipping x-axis (the enviroment is not quite symmetric
    so leave last column alone) """
    
    x_extent = observation.shape[1]
    x_reflect_almost = lambda x: (x if x == x_extent-1 else
                                  x_extent-1 - x)

    obs_copy = deepcopy(observation)
    for x in range(x_extent):
        observation[:, x, :] = obs_copy[:, x_reflect_almost(x), :]

    return observation
##


def _swap_bat_colours(observation):
    """ swap colours of bats """
    
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

