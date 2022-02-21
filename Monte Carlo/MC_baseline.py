## RL MC with exploring starts for gym Blackjack (gym 0.22.0)
import gym
import time
import random
import numpy as np
from gym.envs.toy_text.blackjack import BlackjackEnv
env = BlackjackEnv()
env_new=env.reset()
print(env_new)
def greedy_policy(Q_table,obs):#obs是一个元组，为3维， Q_table 为4维
    index=(obs[0],obs[1],int(obs[2]))
    return np.argmax(Q_table[index])
def MC_evaluation_update(data,Q_table,iteration_Q):#这里为了避免记录每一幕数据来求平均G，采取了增量式，用iteration_Q记录迭代次数。
    G=0
    gamma=1
    obs,rewards,actions=data['obs'],data['rewards'],data['actions']
    for i in range(len(obs)-1,-1,-1): #采取是倒序计算，更方便
        G= G*gamma + rewards[i]
        if obs[i] in obs[:i] and actions[i] in actions[:i]: #首次访问 ，只记录第一次
            continue
        index= (obs[i][0],obs[i][1],obs[i][2],actions[i]) #操作的索引值
        iteration_Q[index]+=1 #迭代次数加1
        Q_table[index]+=(1/iteration_Q[index])*( Q_table[index]-G) #迭代式 Q<-Q+1/k(Q-G)
    return Q_table,iteration_Q
def sample_data(env,Q_table,random_policy=False):
    obs,reward,actions=[],[],[]
    observation = env.reset()
    obs.append(observation)
    done = False
    while not done:
        if random_policy:
            action=random.randint(0,1)
        else:
            action = greedy_policy(Q_table,observation)
        observation,r,done,_ = env.step(action)
        if not done:
            obs.append(observation)
        actions.append(action)
        reward.append(r)
    return {'obs':obs,
            'actions':actions,
            'rewards':reward}
def training(env,iteration_time=50000):
    obs_space=env.observation_space
    action_space=env.action_space
    Q_shape = [item.n for item in obs_space]+[action_space.n]
    Q_table = np.zeros(Q_shape)
    iteration_Q = np.zeros(Q_shape)
    print(Q_table.shape)
    for _ in range(iteration_time):
        data = sample_data(env,Q_table)
        Q_table,iteration_Q = MC_evaluation_update(data,Q_table,iteration_Q)
    return Q_table
def testing(Q_table,random_policy=False,testing_time=20000):
    win,draw,lose=0,0,0
    for i in range(testing_time):
        data = sample_data(env,Q_table,random_policy)
        if data['rewards'][-1]>0:
            win+=1
        elif data['rewards'][-1]<0:
            lose+=1
        else:
            draw+=1
    print('win: '+str(win)+' lose: '+str(lose)+' draw game: ' +str(draw))
    print('win_rate: '+str(win/testing_time)[:4]+' lose_rate: '+str(lose/testing_time)[:4]+' draw_rate: '+str(draw/testing_time)[:4])
Q_table=training(env,iteration_time=400000)
testing(Q_table)
testing(Q_table,random_policy=True)
