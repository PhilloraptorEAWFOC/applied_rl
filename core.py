import gym
import numpy as np
import pandas as pd


class NoisedObservationWrapper(gym.ObservationWrapper):
    # encapsulate environment with this wrapper and use env just like before:
    # env = NoisedObservationWrapper(gym.make("CartPole-v0"))
    # for PPO2:
    # from stable_baselines.common import make_vec_env
    # env = make_vec_env(NoisedObservationWrapper(gym.make("CartPole-v0")))
    
    def __init__(self, env, std_dev=0.1, mean=0):
        super(NoisedObservationWrapper, self).__init__(env)
        self.std_dev = std_dev
        self.mean = mean
      
    def observation(self, observation):
        # Modify observation here, e.g. just add noise at certain angles etc.
        # for now we add noise with mu=0 and std=0.1 on all observations
        return observation + np.random.normal(self.mean, self.std_dev, 4)


def sample_data(env, episodes=200, save=False):
    env.np_random.seed(0)

    CART_POS="cartPos"
    CART_VEL="cartVel"
    PEND_POS="pendPos"
    PEND_VEL="pendVel"
    EPISODE="episode"
    STEP="step"
    ACTION="action"

    # create empty Pandas dataset
    d = {
        CART_POS:[], CART_VEL:[], 
        PEND_POS:[], PEND_VEL:[],
        EPISODE:[], STEP:[], ACTION:[]
        }
    df = pd.DataFrame(data=d)

    # sample data
    for episode in range(episodes):
        print ("Start of episode %d" % episode)
        obs = env.reset()
        step = 0
        done = False
        
        while step < 500 and not done:
            step += 1
            action = env.action_space.sample()
            
            df = df.append ({CART_POS:obs[0], CART_VEL:obs[1], 
                            PEND_POS:obs[2], PEND_VEL:obs[3],
                            EPISODE:episode, STEP:step, ACTION:action}, ignore_index=True)
            
            obs, reward, done, _ = env.step(action)
    
    if save:
        store = pd.HDFStore('sample_data.h5')
        store['df'] = df
    
    return df