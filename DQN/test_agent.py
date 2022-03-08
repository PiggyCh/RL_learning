import math
import random
import time
import flappy_bird_gym
import cv2
import torch
from utils.utils import process_img
import numpy as np
from algorithm.DQN_agent import DQN
from arguments import arguments
import matplotlib.pyplot as plt
env = flappy_bird_gym.make("FlappyBird-rgb-v0")
arg = arguments()
agent=DQN(env,arg)
def test_performance():
    reward_list = []
    for i in range(10000):
        obs = process_img(env.reset())
        obs = np.expand_dims(obs, axis=0)
        obs = np.repeat(obs, 4, axis=0)
        done = False
        reward = 0
        while not done:
            env.render()
            action = agent.greedy_action(obs)
            transition = env.step(action)
            next_obs = np.expand_dims(process_img(transition[0]),axis=0)
            obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
            reward += transition[1]
            done = transition[2]
        reward_list.append(reward)
    return (sum(reward_list)/arg.test_episodes)-101

def load_state():
    modelpath = 'model/model33000.pkl'
    state_file = torch.load(modelpath)
    agent.Net.load_state_dict(state_file)
    agent.targetNet.load_state_dict(state_file)
if __name__ == '__main__':
    load_state()
    test_performance()