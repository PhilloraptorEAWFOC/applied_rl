import gym
import numpy as np


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