import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt
class args():
    def __init__(self):
        self.act_dim =env.action_space.n
        self.obs_dim =env.observation_space.shape[0]
        self.gamma = 0.99
        self.lr  = 1e-2
        self.train_episodes = 10000
        self.test_interval= 2
        self.test_epsiodes= 10
env=gym.make('CartPole-v1')
env.seed(1)
arg = args()
def test_performance(agent):
    total_reward=[]
    for test_episode in range(arg.test_epsiodes):
        o = env.reset()
        done = False
        reward,action,obs,next_obs,log_prob_list=[],[],[o],[],[]
        while not done:
            a = agent.get_greedy_action(o)
            new_o,r,done,_ = env.step(a)
            reward.append(r)
            action.append(a)
            obs.append(o)
            next_obs.append(new_o)
            o = new_o
        total_reward.append(sum(reward))
    return  sum(total_reward)/len(total_reward)
def training(arg):
    agent = policy_gradient(arg)
    reward_test_list = []
    for train_episode in range(arg.train_episodes):
        o = env.reset()
        done = False
        reward,action,obs,next_obs,log_prob_list=[],[],[o],[],[]
        while not done:
            a = agent.get_action(o)
            new_o,r,done,_ = env.step(a)
            reward.append(r)
            action.append(a)
            obs.append(o)
            next_obs.append(new_o)
            o = new_o
        agent.update({
            'action': action,
            'obs': obs,
            'next_obs': next_obs,
            'reward': reward
        })
        # # update

        if train_episode % arg.test_interval==1:
            reward_test_list.append(test_performance(agent))
            print('Train_episodes: '+str(train_episode) +' average_reward: '+str(reward_test_list[-1]))
            if reward_test_list[-1]==500:
                plt.plot([5*i for i in range(len(reward_test_list))],reward_test_list)
                plt.xlabel("training number")
                plt.ylabel("score")
                plt.show()
class agent_net(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(agent_net, self).__init__()
        self.FC1 = nn.Linear(obs_dim,64)
        self.FC3 = nn.Linear(64,act_dim) #离散动作logits
        self.Relu =nn.ReLU()
        self.softmax =nn.Softmax()
    def forward(self,x):  #前向传播
        x = self.Relu(self.FC1(x))
        #x = self.Relu(self.FC2(x))
        x = self.softmax(self.FC3(x))
        return x

class policy_gradient():
    def __init__(self,args):
        self.args = args
        self.a_Net = agent_net(args.obs_dim,args.act_dim)
        self.optimizer = torch.optim.Adam(self.a_Net.parameters(),lr=args.lr)
        self.loss = torch.nn.CrossEntropyLoss()
    def get_action(self,obs): #输出动作
        obs = torch.tensor(obs,dtype=torch.float32)
        logits = self.a_Net(obs)
        dist = Categorical(logits)
        action = dist.sample()
        return action.detach().cpu().numpy()
    def get_greedy_action(self,obs):
        obs = torch.tensor(obs,dtype=torch.float32)
        logits = self.a_Net(obs)
        action = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        return action
    def evaluate_actions(self, obs, action):
        actor_features = self.a_Net(obs)
        dist = Categorical(actor_features)
        action_log_probs = dist.log_prob(action)
        return -action_log_probs
    def update(self,data): #更新agent
        a,obs,next_obs,r = data['action'],data['obs'],data['next_obs'],data['reward']
        a = torch.tensor(np.array(a),dtype=torch.int32)
        obs = torch.tensor(obs[:-1], dtype=torch.float32)
        log_prob = self.evaluate_actions(obs,a)
        r_tmp = 0
        for i in range(len(r)-1,-1,-1):
            r[i] = r_tmp*self.args.gamma+r[i]
            r_tmp = r[i]
        r = (r-np.mean(r))/np.std(r)
        r = torch.tensor(r,dtype=torch.int32)
        loss = (log_prob*r).mean()
        # dist =
        # torch.zero
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    arg =args()
    training(arg)
