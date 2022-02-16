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
import logging
from functools import partial

import numpy as np


# noinspection SpellCheckingInspection,PyPep8Naming,DuplicatedCode
class IBHP:
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING)

    def __init__(self, n_sample=100):
        # params for generating data
        self.n_sample = n_sample
        self.L = None  # base kernel的数量
        self.D = None  # 每个文档的长度
        self.S = None  # 词典的总词数（下面用每个词的index代替）
        self.w_0 = None  # 文档的主题分布(dirichlet)参数
        self.w = None  # base kernel的权重
        self.v_0 = None  # 主题的词分布(dirichlet)参数
        self.v = None  # 词分布
        self.lambda_k_array_mat = None  # factor rate
        self.beta = None  # base kernel初始参数
        self.tau = None  # base kernel初始参数
        self.c = None  # 主题向量
        self.timestamp_array = None  # 存储每个样本时间戳的向量
        self.kappa_n = None  # factor kernel vec
        self.gamma = None  # base kernel vec
        self.T = None  # sample text vector
        self.K = None  # 主题数
        self.lambda0 = None
        self.lambda_tn_array = None
        self.s = None

    @staticmethod
    def base_kernel_l(delta, beta, tao):
        base_kernel_l = beta * np.exp(- delta / tao)
        return base_kernel_l

    @staticmethod
    def generate_c(p_k):
        if np.isnan(p_k):
            logging.warning('p_k is nan')
            p_k = 0
        c_old = np.random.binomial(n=1, p=p_k)
        return c_old

    def calculate_lambda_k(self, n):
        if n == 1:
            self.lambda_k_array_mat = self.w.T @ self.beta
            self.lambda_k_array_mat = self.lambda_k_array_mat.reshape(-1, self.lambda_k_array_mat.shape[0])
        elif n >= 2:
            pass

    def generate_first_event(self):
        # 参数初始化
        self.L = 3  # base kernel的数量
        self.D = 20  # 每个文档的长度
        self.S = 1000  # 词典的总词数（下面用每个词的index代替）
        self.K = 0
        self.lambda0 = 2
        self.lambda_k_array_mat = None
        self.tau = np.array([0.3, 0.2, 0.1])  # base kernel初始参数
        self.beta = np.array([1, 2, 3])  # base kernel初始参数
        self.w_0 = np.array([1 / 3, 1 / 3, 1 / 3])  # base kernel的权重分布(dirichlet)参数
        self.v_0 = np.array([1 / 1000] * self.S)  # 主题的词分布(dirichlet)参数

        # Generate the First Event
        while self.K == 0:
            self.K = np.random.poisson(self.lambda0, 1)[0]
            logging.info(f'n=1时主题数量为0，重新生成主题数量')
        logging.info(f'n=1时主题数量: {self.K}')
        # 初始化c矩阵
        self.c = np.ones((1, self.K))
        logging.info(f'n=1时主题出现情况: {self.c}')

        # sample w_k
        self.w = np.random.dirichlet(self.w_0, self.K).T
        logging.info(f'n=1时base kernel的权重：{self.w}')

        # calculate each base kernel
        base_kernel = np.vectorize(partial(self.base_kernel_l, 0))  # event 1的delta参数为0
        self.gamma = base_kernel(self.beta, self.tau)
        logging.info(f'n=1时的base kernel：{self.gamma}')

        # calculate factor kernel
        self.kappa_n = self.w.T @ self.beta
        logging.info(f'n=1时的factor kernel：{self.kappa_n}')

        # sample v_k
        self.v = np.random.dirichlet(self.v_0, self.K).T

        # sample t_1
        self.generate_timestamp(1)
        logging.info(f'n=1时的时间戳：{self.timestamp_array}')

        # sample T_1
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa_n != 0)[:, 0]]) / np.count_nonzero(
            self.kappa_n)
        self.T = np.random.multinomial(self.D, multi_dist_prob, size=1)
        logging.info(f'n=1时的生成数据：{self.T}')

        # calculate lambda_1
        self.calculate_lambda_k(n=1)
        assert isinstance(self.lambda_k_array_mat, np.ndarray)
        self.lambda_tn_array = np.array(np.sum(self.lambda_k_array_mat))

    def generate_following_event(self, n: int):
        # 计算从已有K个主题中生成c的概率
        p = self.lambda_k_array_mat[-1] / ((self.lambda0 / self.K) + self.lambda_k_array_mat[-1])
        logging.info(f'n={n}时从已有K个主题中生成c的概率: {p}')

        # 生成已有K个主题的c向量
        generate_old_c = np.vectorize(self.generate_c)
        c_old = generate_old_c(p)
        while np.all(c_old == 0):
            logging.info(f'n={n}时主题出现情况全为0，重新采样')
            c_old = generate_old_c(p)

        # sample K+
        K_plus = np.random.poisson(self.lambda0 / (self.lambda0 + np.sum(self.lambda_k_array_mat[-1])), 1)[0]

        # update K
        self.K = self.K + K_plus

        # generate new c vec if K_plus > 0
        if K_plus:
            c_new = np.ones(K_plus)
            c = np.hstack((c_old, c_new))
            self.c = np.hstack((self.c, np.zeros((self.c.shape[0], K_plus))))  # 将已有的c矩阵用0补全
            self.c = np.vstack((self.c, c))  # 将新样本的c向量加入c矩阵

            new_w = np.random.dirichlet(self.w_0, K_plus).T
            self.w = np.hstack((self.w, new_w))

            new_v = np.random.dirichlet(self.v_0, K_plus).T
            self.v = np.hstack((self.v, new_v))

            # update kappa vector
            new_kappa = new_w.T @ self.beta
            self.kappa_n = self.w[:, c_old.shape[0]].T @ self.beta * c_old
            self.kappa_n = np.hstack((self.kappa_n, new_kappa))
        else:
            self.c = np.vstack((self.c, c_old))
            self.kappa_n = self.w.T @ self.beta * c_old

        logging.info(f'n={n}时主题出现情况：{self.c}')
        logging.info(f'n={n}时factor kernel：{self.kappa_n}')

        # calculate lambda(t_n)
        # self.lambda_tn_array = np.append(self.lambda_tn_array, lambda_tn)

        # sample t_n
        res_code = self.generate_timestamp(n)

        # sample T_n
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa_n[-1, :] != 0)[:, 0]]) / \
                          np.count_nonzero(self.kappa_n[-1, :])
        T_n = np.random.multinomial(self.D, multi_dist_prob, size=1)
        self.T = np.append(self.T, T_n, axis=0)  # 更新生成文本的矩阵
        logging.info(f'n={n}时的生成数据：{T_n}')

    def generate_timestamp(self, n: int):
        if n == 1:
            lambda_star = self.lambda0
            u = np.random.uniform(0, 1, 1)[0]
            self.s = (- 1 / lambda_star) * np.log(u)
            if n <= self.n_sample:
                self.timestamp_array = np.array([self.s])
            else:
                return 'exit'
        elif n >= 2:
            FLAG = 'notpass'
            non_zero_kappa_idx = np.argwhere(self.kappa_n[n - 1] != 0)[:, 0]
            lambda_star = self.lambda_tn_array[n - 2] + np.sum(
                np.einsum('ij,i->j', self.w[:, non_zero_kappa_idx], self.beta))
            while FLAG is 'notpass':
                u = np.random.uniform(0, 1, 1)[0]
                self.s = self.s - (1 / lambda_star) * np.log(u)
                if n > self.n_sample:
                    return 'exit'
                # rejection test
                # calculate lambda(s)
                delta_s = self.s - self.timestamp_array[self.timestamp_array < self.s]
                t_div_tau_func = np.vectorize(np.divide, signature='(),(n)->(n)')
                beta_times_tau_func = np.vectorize(lambda x, y: x * y, signature='(n),(n)->(n)')
                gamma_array = beta_times_tau_func(self.beta, np.exp(- t_div_tau_func(delta_s, self.tau)))
                numerator = (self.w[:, non_zero_kappa_idx].T @ gamma_array.T).T
                denominator = np.count_nonzero(self.kappa_n[: n, :], axis=1).reshape(-1, 1)
                lambda_s = np.einsum('ij->', numerator / denominator)

                d = np.random.uniform(0, 1, 1)[0]
                if d <= lambda_s / lambda_star:
                    self.timestamp_array = np.append(self.timestamp_array, self.s)
                    FLAG = 'pass'
                else:
                    lambda_star = lambda_s

    def generate_data(self):
        self.generate_first_event()
        for i in np.arange(1, self.n_sample):
            self.generate_following_event(n=i + 1)


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    ibhp_ins = IBHP(n_sample=100)
    ibhp_ins.generate_data()
