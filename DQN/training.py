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
    for i in range(arg.test_episodes):
        obs = process_img(env.reset())
        obs = np.expand_dims(obs, axis=0)
        obs = np.repeat(obs, 4, axis=0)
        done = False
        reward = 0
        while not done:
            action = agent.greedy_action(obs)
            transition = env.step(action)
            next_obs = np.expand_dims(process_img(transition[0]),axis=0)
            obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
            reward += transition[1]
            done = transition[2]
        reward_list.append(reward)
    return (sum(reward_list)/arg.test_episodes)-101
def load_state():
    modelpath = 'model/model4200.pkl'
    state_file = torch.load(modelpath)
    agent.Net.load_state_dict(state_file)
    agent.targetNet.load_state_dict(state_file)
def training():
    reward_graph_data=[]
    #plt.ion()
    fig, ax = plt.subplots()
    for i in range(arg.episodes):
        obs = process_img(env.reset())
        obs = np.expand_dims(obs, axis=0)
        obs = np.repeat(obs, 4, axis=0)
        done = False
        transition_store = {
            'obs': [],
            'next_obs': [],
            'action': [],
            'reward': [],
            'done': []
        }
        if i%10000==0 and i>100:
            arg.epsilon/=math.sqrt(10)
        while not done:
            action = agent.get_action(obs)
            if i < 300:
                action = random.randint(0,1)
            transition = env.step(action)
            next_obs = np.expand_dims(process_img(transition[0]),axis=0)
            next_obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
            reward = transition[1]
            done = transition[2]
            if done:
                reward -= 101
            #next_obs = np.stack(next_obs, axis=0)
            # plt.imshow(obs[0], cmap="gray",interpolation='nearest')
            # plt.show()
            transition_store['obs'].append(obs)
            transition_store['next_obs'].append(next_obs)
            transition_store['reward'].append(reward)
            transition_store['done'].append(done)
            transition_store['action'].append(action)
            obs = next_obs
        if sum(transition_store['reward'])>0:
            print('breakthourgh!'+str(sum(transition_store['reward'])))
        agent.Buffer.store_data(transition_store, len(transition_store['obs']))
        if agent.Buffer.ptr > 500:  # 保证前期存有一定的样本
            agent.update(agent.Buffer.sample(arg.updatebatch))
        if i % 50 == 0:
            if i% 300 ==0:
                torch.save(agent.Net.state_dict(),'model/model'+str(i)+'.pkl')
            average_r = test_performance()
            print('iteration episodes: '+str(i)+' test average reward: '+str(average_r))
            reward_graph_data.append(average_r)
            ax.plot(reward_graph_data, 'g-', label='reward')
            plt.savefig('graph.jpg')
            #plt.clf()

if __name__ == '__main__':
    load_state()
    test_performance()
    #training()
