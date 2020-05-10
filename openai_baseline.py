
from baselines.a2c.a2c import learn
from baselines.common.vec_env import DummyVecEnv
import retro

cache_path = 'pong_a2c_custom_baseline'

def get_env():
    return retro.make(game='Pong-Atari2600', players=1)
##

def train_save_baseline():
    env = get_env()
    model = learn('cnn_small',
                  DummyVecEnv([lambda: env]),
                  total_timesteps=int(80e6))
    model.save(save_path=cache_path)
##


def load_baseline():
    env = get_env()
    model = learn('cnn_small',
                        DummyVecEnv([lambda: env]),
                        total_timesteps=0)
    model.load(load_path=cache_path)
    env.close()
    return model
##


if __name__=='__main__':
    train_save_baseline()
    
