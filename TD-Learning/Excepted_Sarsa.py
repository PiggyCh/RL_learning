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
    Q_table = np.zeros([obs_space,action_space])
    max_val=-100000
    for training_time in range(arg.iteration_time):
        s = env.reset()
        a = epsilon_policy(s, arg.epsilon, Q_table)
        done = False
        steps=0
        while not done:
            s_new, r, done, _ = env.step(a)
            steps+=1
            if steps>200:
                done=True
            if not done:
                max_a = np.argmax(Q_table[s_new]) #根据当前观察s选出最优的动作以进行更新
                prob_pi=[arg.epsilon/action_space for _ in range(action_space)]
                prob_pi[max_a]+=1-arg.epsilon # 得到输出概率
                Q_target = np.dot(Q_table[s_new],np.array(prob_pi))#计算TD目标
                Q_table[s, a] += arg.alpha*(r + arg.gamma * Q_target -Q_table[s, a]) #更新公式
                s, a = s_new, epsilon_policy(s, arg.epsilon, Q_table) #进行下一循环
            else:
                Q_table[s, a] += arg.alpha * (r - Q_table[s, a])
        if training_time>1 and training_time %1 ==0:
            reward_test = testing(Q_table,arg)
            if reward_test>max_val:
                print(reward_test)
                max_val=reward_test
                Q_best = Q_table[:]
            if training_time % 100 == 0:
                print(training_time)
            plot_fig(reward_training)
    return Q_best


arg=arguments()
Q_best = training(env,arg)
reward_test = testing_trajectory(Q_best, arg)
print(reward_test)
