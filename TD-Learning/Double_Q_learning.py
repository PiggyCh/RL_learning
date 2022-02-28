import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv
env = CliffWalkingEnv()
# print(env.observation_space.n)
# print(env.action_space.n)
obs_space = env.observation_space.n
action_space = env.action_space.n
reward_training=[]

from matplotlib.patches import Circle
import math
plt.close()  # clf() # 清图 cla() # 清坐标轴 close() # 关窗口
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")  # 设置图像显示的时候XY轴比例
plt.grid(True)  # 添加网格
#plt.ion()  # interactive mode on


class arguments():
    def __init__(self):
        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma= 0.99
        self.iteration_time = 20000
        self.testing_time = 40

def plot_fig(reward_training):
    plt.cla()
    obsX = [i for i in range(len(reward_training))]
    obsY = reward_training
    ax.set(xlim=(0, len(obsX)))
    ax.plot(obsX, obsY, c='b')  # 散点图
    # ax.lines.pop(1) 删除轨迹
    # 下面的图,两船的距离
    plt.pause(0.01)


def epsilon_policy(obs,epsilon,Q_table):
    if random.random()<epsilon:
        return random.randint(0,3)
    else:
        return np.argmax(Q_table[obs])
def testing(Q_table,arg):
    r_list=[]
    for i in range(arg.testing_time):
        s = env.reset()
        done =False
        r_total=0
        steps=0
        while not done:
            a = epsilon_policy(s,arg.epsilon,Q_table)
            s,r,done,_ = env.step(a)
            steps+=1
            if steps>200:
                done=True
            r_total+=r
        r_list.append(r_total)
    reward_training.append(sum(r_list)/len(r_list))
    return sum(r_list)/len(r_list)

def testing_trajectory(Q_table,arg):
    r_list=[]
    for i in range(1):
        s = env.reset()
        done =False
        r_total=0
        steps=0
        while not done:
            a = epsilon_policy(s,arg.epsilon,Q_table)
            env.render()
            s,r,done,_ = env.step(a)
            steps+=1
            if steps>200:
                done=True
            r_total+=r
        r_list.append(r_total)
    reward_training.append(sum(r_list)/len(r_list))
    return sum(r_list)/len(r_list)

def training(env,arg):
    Q_table_1 = np.zeros([obs_space,action_space]) #建立Q1表
    Q_table_2 = np.zeros([obs_space,action_space]) #建立Q1表
    max_val=-100000
    for training_time in range(arg.iteration_time): #训练幕数
        s = env.reset()
        a = epsilon_policy(s, arg.epsilon, Q_table_1+Q_table_2)  #策略输出动作
        done = False
        steps=0
        while not done:
            s_new, r, done, _ = env.step(a)
            steps+=1
            if steps>200: #步数超过200终止
                done=True
            if not done:
                if random.random()<0.5: #以0.5的概率选择Q2
                    max_a = np.argmax(Q_table_2[s_new])
                    Q_table_2[s, a] += arg.alpha*(r + arg.gamma * Q_table_1[s_new,max_a] -Q_table_2[s, a]) #更新公式
                else: #以0.5的概率选择Q2
                    max_a = np.argmax(Q_table_1[s_new])
                    Q_table_1[s, a] += arg.alpha*(r + arg.gamma * Q_table_2[s_new,max_a] -Q_table_1[s, a]) #更新公式
                s, a = s_new, epsilon_policy(s, arg.epsilon, Q_table_1+Q_table_2) #进行下一循环
            else:
                Q_table_1[s, a] += arg.alpha * (r - Q_table_1[s, a])
        if training_time>18000 and training_time %1 ==0:
            reward_test = testing(Q_table_1+Q_table_2,arg)
            if reward_test>max_val:
                print(reward_test)
                max_val=reward_test
                Q_best = (Q_table_1+Q_table_2)[:]
            if training_time % 100 == 0:
                print(training_time)
            plot_fig(reward_training)
    return  Q_best


arg=arguments()
Q_best = training(env,arg)
reward_test = testing_trajectory(Q_best, arg)
print(reward_test)
