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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


# noinspection SpellCheckingInspection,PyPep8Naming,DuplicatedCode
class IBHP:
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING)

    def __init__(self, n_sample=100):
        # params for generating data
        self.lambda_k_array_mat = None
        self.n_sample = n_sample
        self.L = None  # base kernel的数量
        self.D = None  # 每个文档的长度
        self.S = None  # 词典的总词数（下面用每个词的index代替）
        self.w_0 = None  # 文档的主题分布(dirichlet)参数
        self.w = None  # base kernel的权重
        self.v_0 = None  # 主题的词分布(dirichlet)参数
        self.v = None  # 词分布
        self.lambda_k_array = None  # factor rate
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
        """
        calculate lambda_k array for each event
        :param n:
        :return:
        """
        if n == 1:
            self.lambda_k_array = self.w.T @ self.beta
        elif n >= 2:
            # kappa_n_nonzero_index = np.argwhere(self.kappa_n != 0)[:, 0]
            delta_t_array = self.timestamp_array[-1] - self.timestamp_array

            base_kernel_for_delta_t_vec = np.vectorize(self.base_kernel_l, signature='(n),(),()->(n)')
            base_kernel_mat = base_kernel_for_delta_t_vec(delta_t_array, self.beta, self.tau).T  # t_i for each row
            kappa_history = np.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c
            kappa_history_count = np.count_nonzero(kappa_history, axis=1).reshape(-1, 1)
            self.lambda_k_array = np.sum(np.divide(kappa_history, kappa_history_count), axis=0)

    def generate_first_event(self):
        # 参数初始化
        self.L = 3  # base kernel的数量
        self.D = 20  # 每个文档的长度
        self.S = 1000  # 词典的总词数（下面用每个词的index代替）
        self.K = 0
        self.lambda0 = 2
        self.lambda_k_array = None
        self.tau = np.array([0.3, 0.2, 0.1])  # base kernel初始参数
        self.beta = np.array([1, 2, 3])  # base kernel初始参数
        self.w_0 = np.array([1 / 3, 1 / 3, 1 / 3])  # base kernel的权重分布(dirichlet)参数
        self.v_0 = np.array([1 / self.S] * self.S)  # 主题的词分布(dirichlet)参数

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
        logging.info(f'n=1时w：{self.w}')

        # calculate each base kernel
        base_kernel = np.vectorize(partial(self.base_kernel_l, 0))  # event 1的delta参数为0
        self.gamma = base_kernel(self.beta, self.tau)
        logging.info(f'n=1时的base kernel：{self.gamma}')

        # calculate factor kernel
        self.kappa_n = self.w.T @ self.beta
        logging.info(f'n=1时的kappa_n：{self.kappa_n}')

        # sample v_k
        self.v = np.random.dirichlet(self.v_0, self.K).T

        # sample t_1
        self.generate_timestamp(1)
        logging.info(f'n=1时的时间戳：{self.timestamp_array}')

        # sample T_1
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa_n != 0)[:, 0]]) / np.count_nonzero(
            self.kappa_n)
        self.T = np.random.multinomial(self.D, multi_dist_prob, size=1)
        # logging.info(f'n=1时的生成数据：{self.T}')

        # calculate lambda_1
        self.calculate_lambda_k(n=1)
        assert isinstance(self.lambda_k_array, np.ndarray)
        self.lambda_tn_array = np.array([np.sum(self.lambda_k_array) + self.lambda0])
        logging.info(f'n=1时的lambda_tn_array：{self.lambda_tn_array}\n')

        self.collect_factor_intensity(1)

    def generate_following_event(self, n: int):
        # 计算从已有K个主题中生成c的概率
        p = self.lambda_k_array / ((self.lambda0 / self.K) + self.lambda_k_array)
        logging.info(f'n={n}时从已有K个主题中生成c的概率: {p}')

        # 生成已有K个主题的c向量
        generate_old_c = np.vectorize(self.generate_c)
        c_old = generate_old_c(p)
        while np.all(c_old == 0):
            logging.info(f'n={n}时主题出现情况全为0，重新采样')
            c_old = generate_old_c(p)

        # sample K+
        K_plus = np.random.poisson(self.lambda0 / (self.lambda0 + np.sum(self.lambda_k_array)), 1)[0]

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
            self.kappa_n = self.w[:, : c_old.shape[0]].T @ self.beta * c_old
            self.kappa_n = np.hstack((self.kappa_n, new_kappa))
        else:
            self.c = np.vstack((self.c, c_old))
            self.kappa_n = self.w.T @ self.beta * c_old

        logging.info(f'n={n}时主题出现情况：{self.c}')
        logging.info(f'n={n}时kappa_n：{self.kappa_n}')

        # sample t_n
        self.generate_timestamp(n)
        logging.info(f'n={n}时的时间戳：{self.timestamp_array}')

        # sample T_n
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa_n != 0)[:, 0]]) / \
                          np.count_nonzero(self.kappa_n)
        T_n = np.random.multinomial(self.D, multi_dist_prob, size=1)
        self.T = np.append(self.T, T_n, axis=0)  # 更新生成文本的矩阵
        # logging.info(f'n={n}时的生成数据：{T_n}\n')

        # update lambda_k_matrix
        self.calculate_lambda_k(n)

        # calculate lambda(t_n)
        kappa_n_nonzero_index = np.argwhere(self.kappa_n != 0)[:, 0]
        self.lambda_tn_array = np.append(self.lambda_tn_array,
                                         np.sum(self.lambda_k_array[kappa_n_nonzero_index]) + self.lambda0)
        logging.info(f'n={n}时的lambda_tn_array：{self.lambda_tn_array}\n')

        self.collect_factor_intensity(n)

    def generate_timestamp(self, n: int):
        if n == 1:
            lambda_star = self.lambda0
            u = np.random.uniform(0, 1, 1)[0]
            self.s = np.random.exponential(lambda_star)
            self.timestamp_array = np.array([self.s])
        elif n >= 2:
            FLAG = 'notpass'
            # update maximum intensity
            nonzero_index_kappa_last = np.argwhere(self.kappa_n != 0)[:, 0]  # event(n-1)
            Y = np.sum(self.w[:, nonzero_index_kappa_last].T @ self.beta) / np.count_nonzero(self.kappa_n)
            lambda_star = self.lambda_tn_array[-1] + Y
            while FLAG is 'notpass':
                logging.warning(f'[n={n}]lambda_star: {lambda_star}')
                u = np.random.uniform(0, 1, 1)[0]
                self.s = self.s - (1 / lambda_star) * np.log(u)
                if n > self.n_sample:
                    return 'exit'
                # rejection test
                # calculate lambda(s)
                logging.warning(f'[n={n}]s: {self.s}')
                logging.warning(f'[n={n}]timestamp: {self.timestamp_array}')
                delta_s = self.s - self.timestamp_array[self.timestamp_array <= self.s]
                logging.warning(f'[n={n}]delta_s: {delta_s}')
                base_kernel_for_delta_s_vec = np.vectorize(self.base_kernel_l, signature='(n),(),()->(n)')
                base_kernel_mat = base_kernel_for_delta_s_vec(delta_s, self.beta, self.tau).T
                kappa_s = np.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c[: delta_s.shape[0]]
                logging.warning(f'[n={n}]kappa_s: {kappa_s}')
                kappa_s_nonzero_index = np.argwhere(kappa_s[-1] != 0)[:, 0]
                kappa_s_count = np.count_nonzero(kappa_s, axis=1).reshape(-1, 1)
                kappa_s_count = np.where(kappa_s_count == 0, kappa_s_count, 1)  # Make sure the denominator is not 0
                logging.warning(f'[n={n}]kappa_s_count: {kappa_s_count}')
                lambda_s_array = np.sum(np.divide(kappa_s, kappa_s_count), axis=0)
                logging.warning(f'[n={n}]lambda_s_array: {lambda_s_array}')
                lambda_s = np.sum(lambda_s_array[kappa_s_nonzero_index]) + self.lambda0
                logging.warning(f'[n={n}]lambda_s: {lambda_s}\n')

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

    def plot_intensity_function(self):
        rcParams['font.family'] = 'serif'
        fig, ax = plt.subplots(dpi=500)
        x = np.arange(self.lambda_tn_array.shape[0])
        ax.plot(x, self.lambda_tn_array)
        ax.set_xlabel('n')
        ax.set_ylabel('$\lambda(t_n)$')
        plt.show()

    def collect_factor_intensity(self, n):
        """
        Collect intensity of all factors generated during the simulation
        :param n:
        :return:
        """
        if n == 1:
            self.lambda_k_array_mat = self.lambda_k_array.reshape(1, -1)
        elif n >= 2:
            zero_num = self.lambda_k_array.shape[0] - self.lambda_k_array_mat[-1].shape[0]
            if zero_num:
                self.lambda_k_array_mat = np.hstack((self.lambda_k_array_mat,
                                                     np.zeros((self.lambda_k_array_mat.shape[0], zero_num))))
            self.lambda_k_array_mat = np.vstack((self.lambda_k_array_mat, self.lambda_k_array))

    def plot_each_factor_intensity(self, factor_num):
        """
        :param factor_num: number of random factors need to be plotted
        :return:
        """
        factor_index = np.random.randint(0, self.lambda_k_array_mat.shape[0] - 1, factor_num)
        factor_index = np.unique(factor_index)
        while factor_num - factor_index.shape[0]:
            factor_index = np.append(factor_index,
                                     np.random.randint(0, self.lambda_k_array_mat.shape[0] - 1,
                                                       factor_num - factor_index.shape[0]))
            factor_index = np.unique(factor_index)
        if not factor_num % 3:
            fig, ax = plt.subplots(factor_num // 3, 3, dpi=400)
        else:
            fig, ax = plt.subplots(factor_num // 3 + 1, 3, dpi=400)
        # todo plot factors intensity


if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    ibhp_ins = IBHP(n_sample=100)
    ibhp_ins.generate_data()
    ibhp_ins.plot_intensity_function()
