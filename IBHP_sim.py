#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
@File    :   IBHP_sim.py
@Time    :   2021/11/4 11:10
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import numpy as np
from functools import partial


# noinspection SpellCheckingInspection,PyPep8Naming,DuplicatedCode
class IBHP:

    def __init__(self):
        # params for generating data
        self.L = None  # base kernel的数量
        self.D = None  # 每个文档的长度
        self.S = None  # 词典的总词数（下面用每个词的index代替）
        self.w_0 = None  # 文档的主题分布(dirichlet)参数
        self.w = None  # base kernel的权重
        self.v_0 = None  # 主题的词分布(dirichlet)参数
        self.v = None  # 词分布
        self.Lambda = None  # factor rate
        self.beta = None  # base kernel初始参数
        self.tao = None  # base kernel初始参数
        self.c = None  # 主题向量
        self.timestamp = None  # 存储每个样本时间戳的向量
        self.kappa = None  # factor kernel vec
        self.kappa_all = None  # 记录所有出现过的非零的kappa值
        self.gamma = None  # base kernel vec
        self.T = None  # sample text vector
        self.K = None  # 主题数
        self.Lambda_n_minus_1_arr = None

    @staticmethod
    def base_kernel_l(delta, beta, tao):
        base_kernel_l = beta * np.exp(-delta / tao)
        return base_kernel_l

    @staticmethod
    def generate_c(p_k):
        if np.isnan(p_k):
            print('Warning: p_k is nan')
            p_k = 0
        c_old = np.random.binomial(n=1, p=p_k)
        return c_old

    def generate_first_event(self):
        # 参数初始化
        self.L = 3  # base kernel的数量
        self.D = 20  # 每个文档的长度
        self.S = 100  # 词典的总词数（下面用每个词的index代替）
        self.K = 0
        self.Lambda = np.array([2])  # 泊松过程的抵达率
        self.tao = np.array([0.3, 0.2, 0.1])  # base kernel初始参数
        self.beta = np.array([1, 2, 3])  # base kernel初始参数
        self.w_0 = np.array([1 / 3, 1 / 3, 1 / 3])  # base kernel的权重分布(dirichlet)参数
        self.v_0 = np.array([1 / 100] * self.S)  # 主题的词分布(dirichlet)参数

        # Generate the First Event
        while self.K == 0:
            self.K = np.random.poisson(self.Lambda[-1], 1)[0]
            print(f'n=1时主题数量为0，重新生成主题数量')
        print(f'n=1时主题数量: {self.K}')
        # 初始化c矩阵
        self.c = np.ones((1, self.K))
        print(f'n=1时主题出现情况: {self.c}')

        # sample w_k
        self.w = np.zeros((len(self.w_0), self.K))
        for k in np.arange(self.w.shape[1]):
            self.w[:, k] = np.random.dirichlet(self.w_0)
        print(f'n=1时base kernel的权重：{self.w}')

        # calculate each base kernel
        base_kernel = np.vectorize(partial(self.base_kernel_l, 0))  # 固定delta参数为0
        self.gamma = base_kernel(self.beta, self.tao)
        print(f'n=1时的base kernel：{self.gamma}')

        # calculate factor kernel
        self.kappa = np.zeros((1, self.K))
        for k in np.arange(self.kappa.shape[1]):
            self.kappa[0, k] = self.w[:, k].T @ self.gamma * self.c[0, k]
        # save init kappa
        self.kappa_all = self.kappa.copy().reshape(-1)  # shape=(K, )
        print(f'n=1时的factor kernel all：{self.kappa_all}')
        print(f'n=1时的factor kernel：{self.kappa}')

        # sample v_k
        self.v = np.zeros((self.S, self.K))
        for k in np.arange(self.v.shape[1]):
            self.v[:, k] = np.random.dirichlet(self.v_0)
        print(f'n=1时的词分布：{self.v}')

        # sample t_1 from poisson process, 时间间隔服从参数为 1/Lambda(t) 的指数分布
        time_interval = np.random.exponential(1 / self.Lambda[-1], 1)  # 时间间隔的采样
        self.timestamp = np.array([time_interval])
        print(f'n=1时的时间戳：{self.timestamp}')

        # sample T_1
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa != 0)[:, 0]]) / np.count_nonzero(
            self.kappa, axis=1)[0]
        print(f'n=1时多项分布的参数p: {multi_dist_prob}')
        self.T = np.random.multinomial(self.D, multi_dist_prob, size=1)
        print(f'n=1时的生成数据：{self.T}')

    def generate_following_event(self, n: int):
        if n == 2:
            # 计算factor rate
            self.Lambda_n_minus_1_arr = np.einsum('ij->j', self.kappa / np.count_nonzero(self.kappa, axis=1).
                                                  reshape(self.kappa.shape[0], -1))

        # 计算从已有K个主题中生成c的概率
        p = self.Lambda_n_minus_1_arr / ((self.Lambda[0] / self.K) + self.Lambda_n_minus_1_arr)
        print(f'n={n}时从已有K个主题中生成c的概率: {p}')

        # 生成已有K个主题的c向量
        gen_c = np.vectorize(self.generate_c)
        c_old = gen_c(p)
        while np.all(c_old == 0):
            print(f'n={n}时主题出现情况全为0，重新采样')
            c_old = gen_c(p)

        # sample K+
        K_plus = np.random.poisson(self.Lambda[0] / (self.Lambda[0] + np.sum(self.Lambda_n_minus_1_arr)), 1)[0]

        # update K
        self.K = self.K + K_plus

        # generate new c vec if K_plus > 0
        if K_plus:
            c_new = np.ones(K_plus)
            c = np.hstack((c_old, c_new))
            self.c = np.hstack((self.c, np.zeros((self.c.shape[0], K_plus))))  # 将已有的c矩阵用0补全
            self.c = np.vstack((self.c, c))  # 将新样本的c向量加入c矩阵
        else:
            self.c = np.vstack((self.c, c_old))
        print(f'n={n}时主题出现情况：{self.c}')

        # When K_plus > 0, generate new w_k, and update self.w
        if K_plus:
            new_w = np.zeros((len(self.w_0), K_plus))
            for k in np.arange(K_plus):
                new_w[:, k] = np.random.dirichlet(self.w_0)
            self.w = np.hstack((self.w, new_w))
        print(f'n={n}时base kernel的权重：{self.w}')

        # calculate new kappa (factor kernel)
        if K_plus:
            old_kappa = self.kappa_all * c_old  # 根据c_old更新old kappa
            new_kappa = np.zeros(K_plus)
            for new_k in np.arange(K_plus):
                new_kappa[new_k] = self.w[:, self.K - K_plus + new_k].T @ self.gamma * \
                                   self.c[-1, self.K - K_plus + new_k]

            # 更新kappa_all
            self.kappa_all = np.hstack((self.kappa_all, new_kappa))

            # 更新kappa矩阵
            new_kappa = np.hstack((old_kappa, new_kappa))
            self.kappa = np.hstack((self.kappa, np.zeros((self.kappa.shape[0], K_plus))))  # 将已有的kappa矩阵用0补全
            self.kappa = np.vstack((self.kappa, new_kappa))  # 将新的kappa加入kappa矩阵
        else:
            old_kappa = self.kappa_all * c_old
            self.kappa = np.vstack((self.kappa, old_kappa))
        print(f'n={n}时的factor kernel all：{self.kappa_all}')
        print(f'n={n}时factor kernel的权重：{self.kappa}')

        # When K_plus > 0, generate new v_k, and update self.v
        if K_plus:
            new_v = np.zeros((self.S, K_plus))
            for k in np.arange(K_plus):
                new_v[:, k] = np.random.dirichlet(self.v_0)
            self.v = np.hstack((self.v, new_v))
        print(f'n={n}时的词分布：{self.v}')

        # calculate Lambda(t_n)，这儿有点问题
        non_zero_kappa_idx = np.argwhere(self.kappa[-1, :] != 0)[:, 0]
        Lambda_tn_arr = np.einsum('ij->j', self.kappa / np.count_nonzero(self.kappa, axis=1).
                                  reshape(self.kappa.shape[0], -1))
        self.Lambda_n_minus_1_arr = Lambda_tn_arr  # update Lambda_n_minus_1_arr
        Lambda_tn = np.sum(Lambda_tn_arr[non_zero_kappa_idx])
        self.Lambda = np.append(self.Lambda, Lambda_tn)

        # sample t_n
        time_interval = np.random.exponential(1 / self.Lambda[-1], 1)
        timestamp = self.timestamp[-1] + time_interval
        self.timestamp = np.append(self.timestamp, timestamp)

        # sample T_n
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa[-1, :] != 0)[:, 0]]) / \
                          np.count_nonzero(self.kappa[-1, :])
        print(f'n={n}时多项分布的参数p: {multi_dist_prob}')
        print(f'n={n}时多项分布的参数p求和: {np.sum(multi_dist_prob)}')
        T_n = np.random.multinomial(self.D, multi_dist_prob, size=1)
        self.T = np.append(self.T, T_n, axis=0)  # 更新生成文本的矩阵
        print(f'n={n}时的生成数据：{T_n}')

    def generate_data(self, n_sample: int = 100):
        self.generate_first_event()
        for i in np.arange(1, n_sample):
            self.generate_following_event(n=i + 1)


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    ibhp_ins = IBHP()
    for i in range(100):
        try:
            ibhp_ins.generate_data(n_sample=100)
        except Exception as e:
            print(f'Error: {e}')
            break
