import gym
import random
import numpy as np
env = gym.make('FrozenLake8x8-v0')


def Policy_evaluation(env, policy):
    threshold = 1e-6  # 退出阈值
    gamma = 0.98 # 折扣因子
    value_table = np.zeros(env.observation_space.n)  # 初始化Q表
    evaluation_time=0
    while True:
        evaluation_time+=1
        pre_value_table = value_table[:]  # 记录上次迭代
        for state in range(len(value_table)):  # 计算每个状态
            action = policy[state]
            v=0
            for p, next_s, r, done in env.P[state][action]:
                v += p * (r + gamma * value_table[next_s])
            value_table[state]=v
        if np.sum(np.abs(pre_value_table-value_table)) < threshold and evaluation_time>100:  # 小于则退出循环
            return value_table


def greedy_policy(env, value_table):
    gamma=0.98
    policy = np.zeros(len(value_table))
    for state in range(len(value_table)):
        Q_value = [0] * 4
        for action in range(0, 4):  # 0，1，2，3分别代表四个方向
            for p, next_s, r, done in env.P[state][action]:
                Q_value[action] += p * (r + gamma * value_table[next_s])
        policy[state] = Q_value.index(max(Q_value))
    return policy

def value_iteration():
    threshold = 1e-6  # 退出阈值
    gamma = 0.98 # 折扣因子
    value_table = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)
    iteration_times = 300
    for i in range(iteration_times):
        pre_value_table = value_table[:]  # 记录上次迭代
        for state in range(len(value_table)):  # 计算每个状态
            action = policy[state]
            v=0
            for p, next_s, r, done in env.P[state][action]:
                v += p * (r + gamma * value_table[next_s])
            value_table[state]=v
        policy = greedy_policy(env, value_table)
        if i % 10==0:
            print('iteration:' + str(i))
            print(policy)
            print(value_table)
    return policy
def policy_improvement(env):
    policy = np.zeros(env.observation_space.n)
    iteration_times = 300
    for i in range(iteration_times):
        new_value_table = Policy_evaluation(env, policy)
        new_policy = greedy_policy(env, new_value_table)
        # if np.all(new_policy == policy):
        #     break
        policy = new_policy
        if i % 10==0:
            print('iteration:' + str(i))
            print(policy)
            print(new_value_table)
    return policy


def test_policy(policy, run_time=10):
    success=0
    for _ in range(run_time):
        observation = env.reset()
        while True:
            env.render()  # 环境可视化
            action = int(policy[observation])
            observation,r,done, prob= env.step(action)  # 分别是0:当前状态 1：reward 2：donw 3:prob
            if r>0:
                success+=1
            if done:
                break
    print('success time :'+ str(success)+'/'+str(run_time))

if __name__ == '__main__':
    #policy = policy_improvement(env)
    policy = value_iteration()
    test_policy(policy)
