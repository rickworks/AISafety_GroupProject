
import math
import numpy as np
import tensorflow as tf
import pickle


class DummyRewardPredictor:
    """ Dummy object that has a predict method matching what we need for reward predictor """

    def predict(self, observation, action):
        """ Takes in a single observation and corresponding action and predicts whether human would like this """
        # This ought to be replaced by some trainable model
        return observation.sum() / observation.size()
    ##
##


def calc_prob_o1_greater_o2(rewardPredictor, tragetory1, tragetory2):
    """
        Calculate an estimate to the humans prediction to the choice between two
        tragetories.
        
        Inputs:
          rewardPredictor is an object with a predict(observation, action)
          tragetory1/tragetory2 are arrays of (observation, action) tuples
    """
    rSum1 = sum([rewardPredictor.predict(x[0], x[1]) for x in tragetory1])
    rSum2 = sum([rewardPredictor.predict(y[0], y[1]) for y in tragetory2])
    return math.exp(rSum1) / (math.exp(rSum1) + math.exp(rSum2))
##

def build_pong_cnn_model():
    inp = tf.keras.Input(shape=(160,210,3), batch_size=1)
    x = tf.keras.layers.Conv2D(16, (7,7), 3, activation='relu')(inp)
    x = tf.keras.layers.Conv2D(16, (5,5), 2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return tf.keras.Model(inp, x)
#

def build_pong_r_estimate_model(batch_size=1):
    observation_model = build_pong_cnn_model()
    action_input = tf.keras.Input(1, batch_size)

    # Need to experiment here. These layers are going to decide
    # how observation and action interact, so probably need to
    # be tuned per example.
    x = tf.keras.layers.concatenate([observation_model.outputs[0], action_input], axis=1)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model([observation_model.inputs, action_input], x)
##

def build_cartpole_r_estimate_model(batch_size=1):
    observation_input = tf.keras.Input(shape=(4,1), batch_size=1)
    action_input = tf.keras.Input(1, batch_size)

    # Need to experiment here. These layers are going to decide
    # how observation and action interact, so probably need to
    # be tuned per example.
    x = tf.keras.layers.concatenate([observation_input, action_input], axis=1)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model([observation_input, action_input], x)
##

def build_sum_of_r_model(r_model, tragetory_size):
    """ Builds a model that applies r_model to an entire tragetory and sums the r values """

    o_shape = r_model.inputs[0].shape.as_list()[1:]
    a_shape = r_model.inputs[1].shape.as_list()[1:]
    o_size = int(np.prod(o_shape))
    a_size = int(np.prod(a_shape))
    inp_muxed = tf.keras.layers.Input(shape=(o_size+a_size), batch_size=1, name="demux_input")
    o_inp_demuxed = tf.keras.layers.Lambda(lambda inp_ : inp_[:, 0:o_size], name="demux_extract_channel_0")(inp_muxed)
    o_inp_demuxed = tf.keras.layers.Reshape(o_shape, name="demux_reshape_channel_0")(o_inp_demuxed)
    a_inp_demuxed = tf.keras.layers.Lambda(lambda inp_ : inp_[:, o_size:], name="demux_extract_channel_1")(inp_muxed)
    a_inp_demuxed = tf.keras.layers.Reshape(a_shape, name="demux_reshape_channel_1")(a_inp_demuxed)

    modified_model = tf.keras.Model(inp_muxed, r_model([o_inp_demuxed, a_inp_demuxed]))

    o_inp_vector = tf.keras.Input([tragetory_size] + r_model.inputs[0].shape.as_list()[1:], batch_size=1)
    a_inp_vector = tf.keras.Input([tragetory_size] + r_model.inputs[1].shape.as_list()[1:], batch_size=1)
    o_inp_vector_flattened = tf.keras.layers.Reshape([tragetory_size, -1])(o_inp_vector)
    a_inp_vector_flattened = tf.keras.layers.Reshape([tragetory_size, -1])(a_inp_vector)
    inp_vector_muxed = tf.keras.layers.concatenate([o_inp_vector_flattened, a_inp_vector_flattened], axis=2)
    td = tf.keras.layers.TimeDistributed(modified_model)(inp_vector_muxed)

    out = tf.math.reduce_sum(td, axis=1)
    return tf.keras.Model([o_inp_vector, a_inp_vector], out)
##

def build_p_model(r_model, tragetory_size):
    """ Builds a model for training the given observation_model across tragetorySize frames """

    sum_of_r_model = build_sum_of_r_model(r_model, tragetory_size)

    o_inp_shape = sum_of_r_model.inputs[0].shape.as_list()[1:]
    a_inp_shape = sum_of_r_model.inputs[1].shape.as_list()[1:]
    o_inp_1 = tf.keras.layers.Input(shape=o_inp_shape, batch_size=1)
    a_inp_1 = tf.keras.layers.Input(shape=a_inp_shape, batch_size=1)
    o_inp_2 = tf.keras.layers.Input(shape=o_inp_shape, batch_size=1)
    a_inp_2 = tf.keras.layers.Input(shape=a_inp_shape, batch_size=1)

    exp_sum_of_r_1 = tf.math.exp(sum_of_r_model([o_inp_1, a_inp_1]))
    exp_sum_of_r_2 = tf.math.exp(sum_of_r_model([o_inp_2, a_inp_2]))
    p = tf.math.divide(exp_sum_of_r_1, tf.math.add(exp_sum_of_r_1, exp_sum_of_r_2))
    return tf.keras.Model([o_inp_1, a_inp_1, o_inp_2, a_inp_2], p)
##

def main():
    # Experiment just to prove this works
    filename = "tragetories/1.pkl"
    with open(filename, 'rb') as input:
        tragetory = pickle.load(input)
    ##
    tragetory_size = len(tragetory)
    tragetory = [np.reshape(f, [1, 1, 160, 210, 3]) for f in tragetory]
    tragetory = np.concatenate(tragetory, axis=1)

    actions = np.zeros([1, tragetory.shape[1], 1])

    # out = atari_cnn_model.predict(np.reshape(tragetory[0], [1, 160, 210, 3]))
    r_model = build_pong_r_estimate_model()
    # out = r_model.predict([frame, action])

    # inp = tf.keras.Input(atari_cnn_model.inputs[0].shape.as_list(), batch_size=1)
    # td_layer = tf.keras.layers.TimeDistributed(atari_cnn_model)(inp)
    # p_model = tf.keras.Model(inp, td_layer)
    # out = p_model.predict(tragetory)

    # sum_of_r_model = build_sum_of_r_model(r_model, tragetory_size)
    # out = sum_of_r_model.predict([tragetory, actions])

    p_model = build_p_model(r_model, tragetory_size)
    out = p_model.predict([tragetory, actions, tragetory, actions])
    print(out)
##


if __name__ == "__main__":
    main()
##
