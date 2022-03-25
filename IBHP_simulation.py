#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
@File    :   IBHP_simulation.py
@Time    :   2021/11/4 11:10
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from matplotlib import rcParams
import pyro.distributions as dist


# noinspection SpellCheckingInspection,PyPep8Naming,DuplicatedCode
class IBHP:
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Times New Roman'

    def __init__(self, n_sample=100):
        # params for generating data
        self.lambda_k_array_mat = None
        self.n_sample = n_sample
        self.L = None  # the number of base kernels
        self.D = None  # length of each document
        self.S = None  # The total number of words in the dictionary (replace with the index of each word below)
        self.w_0 = None  # The topic distribution (dirichlet) parameter of the document
        self.w = None  # The weight of the base kernel
        self.v_0 = None  # topic word distribution (dirichlet) parameter
        self.v = None  # word distribution
        self.lambda_k_array = None  # factor rate
        self.beta = None  # base kernel initial parameters
        self.tau = None  # base kernel initial parameters
        self.c = None  # topic vector
        self.timestamp_array = None  # Timestamp for each sample
        self.kappa_n = None  # factor kernel vec
        self.gamma = None  # base kernel vec
        self.text = None  # sample text vector
        self.K = None  # Number of topics
        self.lambda0 = None  # base rate
        self.lambda_tn_array = None  # rate array
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
            delta_t_array = self.timestamp_array[-1] - self.timestamp_array

            base_kernel_for_delta_t_vec = np.vectorize(self.base_kernel_l, signature='(n),(),()->(n)')
            base_kernel_mat = base_kernel_for_delta_t_vec(delta_t_array, self.beta, self.tau).T  # t_i for each row
            kappa_history = np.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c
            kappa_history_count = np.count_nonzero(self.c, axis=1).reshape(-1, 1)
            self.lambda_k_array = np.sum(np.divide(kappa_history, kappa_history_count), axis=0)

    def generate_first_event(self):
        # parameter initialization
        self.L = 3  # numbers of base kernel
        self.D = 20  # length of each document
        self.S = 1000  # The total number of words in the dictionary (replace with the index of each word below)
        self.K = 0
        self.lambda0 = 2
        self.tau = np.array([0.3, 0.2, 0.1])  # decaying parameters
        self.beta = np.array([1, 2, 3])  # initial parameters of base kernel
        self.w_0 = np.array([1 / 3, 1 / 3, 1 / 3])  # The weight distribution (dirichlet) parameter of the base kernel
        self.v_0 = np.array([1 / self.S] * self.S)  # topic word distribution (dirichlet) parameter

        # Generate the First Event
        while self.K == 0:
            self.K = np.random.poisson(self.lambda0, 1)[0]
            logging.info(f'When n=1, the number of topics is 0, and the number of topics is regenerated')
        logging.info(f'The number of topics when n=1: {self.K}')
        # initialize the c matrix
        self.c = np.ones((1, self.K))
        logging.info(f'Topics appeared when n=1: {self.c}')

        # sample w_k
        with pyro.plate('wk_1', self.K):
            self.w = pyro.sample('w_1', dist.Dirichlet(torch.from_numpy(self.w_0)))
        self.w = self.w.numpy().T
        logging.info(f'w when n=1：{self.w}')

        # calculate factor kernel
        self.kappa_n = self.w.T @ self.beta
        logging.info(f'kappa_n when n=1：{self.kappa_n}')

        # sample v_k
        with pyro.plate('vk_1', self.K):
            self.v = pyro.sample('v_1', dist.Dirichlet(torch.from_numpy(self.v_0)))
        self.v = self.v.numpy().T

        # sample t_1
        self.generate_timestamp_by_thinning(1)
        logging.info(f'Timestamp when n=1：{self.timestamp_array}')

        # sample T_1
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa_n != 0)[:, 0]]) / np.count_nonzero(
            self.c[0])
        self.text = np.random.multinomial(self.D, multi_dist_prob, size=1)

        # calculate lambda_1
        self.calculate_lambda_k(n=1)
        assert isinstance(self.lambda_k_array, np.ndarray)
        c_n = np.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_array = np.array([np.sum(self.lambda_k_array[c_n])])
        logging.info(f'lambda tn array when n=1：{self.lambda_tn_array}\n')

        self.collect_factor_intensity(1)

    def generate_following_event(self, n: int):
        # Calculate the probability of generating c from the existing K topics
        p = self.lambda_k_array / ((self.lambda0 / self.K) + self.lambda_k_array)
        logging.info(f'Probability of generating c from existing K topics when n={n}: {p}')

        # Generate c with K topics
        generate_old_c = np.vectorize(self.generate_c)
        c_old = generate_old_c(p)
        while np.all(c_old == 0):
            logging.info(f'When n={n}, the occurrence of the topic is all 0, re-sampling')
            c_old = generate_old_c(p)

        # sample K+
        K_plus = np.random.poisson(self.lambda0 / (self.lambda0 + np.sum(self.lambda_k_array)), 1)[0]

        # update K
        self.K = self.K + K_plus

        # generate new c vec if K_plus > 0
        if K_plus:
            c_new = np.ones(K_plus)
            c = np.hstack((c_old, c_new))
            self.c = np.hstack((self.c, np.zeros((self.c.shape[0], K_plus))))  # Complete the existing c matrix with 0
            self.c = np.vstack((self.c, c))  # Add the c vector of the new sample to the c matrix

            with pyro.plate(f'wk_{n}', K_plus):
                new_w = pyro.sample(f'new_w_{n}', dist.Dirichlet(torch.from_numpy(self.w_0)))
            new_w = new_w.numpy().T
            self.w = np.hstack((self.w, new_w))

            with pyro.plate(f'vk_{n}', K_plus):
                new_v = pyro.sample(f'new_v_{n}', dist.Dirichlet(torch.from_numpy(self.v_0)))
            new_v = new_v.numpy().T
            self.v = np.hstack((self.v, new_v))

            # update kappa vector
            new_kappa = new_w.T @ self.beta
            self.kappa_n = self.w[:, : c_old.shape[0]].T @ self.beta * c_old
            self.kappa_n = np.hstack((self.kappa_n, new_kappa))
        else:
            self.c = np.vstack((self.c, c_old))
            self.kappa_n = self.w.T @ self.beta * c_old

        logging.info(f'The topic appears when n={n}：{self.c}')
        logging.info(f'kappa_n when n={n}：{self.kappa_n}')

        # sample t_n
        self.generate_timestamp_by_thinning(n)
        logging.info(f'Timestamp when n={n}：{self.timestamp_array}')

        # sample T_n
        multi_dist_prob = np.einsum('ij->i', self.v[:, np.argwhere(self.kappa_n != 0)[:, 0]]) / \
                          np.count_nonzero(self.c[n - 1])
        T_n = np.random.multinomial(self.D, multi_dist_prob, size=1)
        self.text = np.append(self.text, T_n, axis=0)  # Update the matrix that generates the text

        # update lambda_k_matrix
        self.calculate_lambda_k(n)

        # calculate lambda(t_n)
        c_n = np.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_array = np.append(self.lambda_tn_array,
                                         np.sum(self.lambda_k_array[c_n]))
        logging.info(f'lambda tn array when n={n}：{self.lambda_tn_array}\n')

        self.collect_factor_intensity(n)

    def generate_timestamp(self, n: int):
        if n == 1:
            lambda_star = self.lambda0
            u = np.random.uniform(0, 1, 1)[0]
            self.s = - (1 / lambda_star) * np.log(u)
            self.timestamp_array = np.array([self.s])
        elif n >= 2:
            FLAG = 'notpass'
            # update maximum intensity
            nonzero_index_kappa_last = np.argwhere(self.c[-1] != 0)[:, 0]  # event index of the last sample
            # Y = np.sum(self.w[:, nonzero_index_kappa_last].T @ self.beta) / np.count_nonzero(self.c[n - 1])
            Y = np.sum(self.w.T @ self.beta) / np.count_nonzero(self.c[n - 1])
            lambda_star = self.lambda_tn_array[-1] + Y
            while FLAG is 'notpass':
                u = np.random.uniform(0, 1, 1)[0]
                self.s = self.s - (1 / lambda_star) * np.log(u)
                if n > self.n_sample:
                    return 'exit'
                # rejection test
                # calculate lambda(s)
                delta_s = self.s - self.timestamp_array[self.timestamp_array <= self.s]
                base_kernel_for_delta_s_vec = np.vectorize(self.base_kernel_l, signature='(n),(),()->(n)')
                base_kernel_mat = base_kernel_for_delta_s_vec(delta_s, self.beta, self.tau).T
                kappa_s = np.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c[: delta_s.shape[0]]
                # kappa_s_nonzero_index = np.argwhere(kappa_s[-1] != 0)[:, 0]
                kappa_s_count = np.count_nonzero(self.c[: delta_s.shape[0]], axis=1).reshape(-1, 1)
                lambda_s_array = np.sum(kappa_s / kappa_s_count, axis=0)
                lambda_s = np.sum(lambda_s_array) + self.lambda0

                d = np.random.uniform(0, 1, 1)[0]
                if d <= lambda_s / lambda_star:
                    self.timestamp_array = np.append(self.timestamp_array, self.s)
                    FLAG = 'pass'
                else:
                    lambda_star = lambda_s

    def generate_timestamp_by_thinning(self, n: int):
        FLAG = None
        while not FLAG:
            if n == 1:
                lambda_star = self.lambda0
                u = np.random.uniform(0, 1)
                candidate_next_timestamp = - (1 / lambda_star) * np.log(u)
                candidate_timestamp_array = np.array([candidate_next_timestamp])
            else:
                lambda_star = self.lambda_tn_array[-1]  # upper bound of Poisson intensity
                u = np.random.uniform(0, 1)
                time_interval = - (1 / lambda_star) * np.log(u)
                candidate_next_timestamp = self.timestamp_array[-1] + time_interval
                candidate_timestamp_array = deepcopy(self.timestamp_array)
                candidate_timestamp_array = np.append(candidate_timestamp_array, candidate_next_timestamp)

            # calculate lambda_(candidate_next_timestamp)
            delta_t = candidate_timestamp_array[-1] - candidate_timestamp_array
            kernel = np.vectorize(self.base_kernel_l, signature='(n),(),()->(n)')
            base_kernel_mat = kernel(delta_t, self.beta, self.tau).T  # (t, l)
            kappa_t = np.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c[: delta_t.shape[0]]
            kappa_t_count = np.count_nonzero(self.c[: delta_t.shape[0]], axis=1).reshape(-1, 1)
            lambda_k_array = np.sum(kappa_t / kappa_t_count, axis=0)
            c_t = np.argwhere(self.c[-1] != 0)[:, 0]
            lambda_t = np.sum(lambda_k_array[c_t])

            # rejection test
            s = np.random.uniform(0, 1)
            if s <= lambda_t / lambda_star:
                if n == 1:
                    self.timestamp_array = np.array([candidate_next_timestamp])
                else:
                    self.timestamp_array = np.append(self.timestamp_array, candidate_next_timestamp)
                FLAG = 'PASS'
            else:
                continue

    def generate_data(self):
        self.generate_first_event()
        for i in np.arange(1, self.n_sample):
            self.generate_following_event(n=i + 1)

    def plot_intensity_function(self):
        fig, ax = plt.subplots(dpi=500)
        ax.plot(self.timestamp_array, self.lambda_tn_array)
        ax.set_xlabel('time')
        ax.set_ylabel(r'$\lambda(t_n)$')
        ax.set_xticks(self.timestamp_array)
        ax.set_xticklabels([])
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
        Randomly select a certain number of topics and draw the rate function of the topic
        :param factor_num: number of random factors need to be plotted
        :return:
        """
        factor_index = np.random.randint(0, self.lambda_k_array_mat.shape[1] - 1, factor_num)
        factor_index = np.unique(factor_index)
        while factor_num - factor_index.shape[0]:
            factor_index = np.append(factor_index,
                                     np.random.randint(0, self.lambda_k_array_mat.shape[1] - 1,
                                                       factor_num - factor_index.shape[0]))
            factor_index = np.unique(factor_index)
        if not factor_num % 3:
            fig, ax = plt.subplots(factor_num // 3, 3, dpi=400)
        else:
            fig, ax = plt.subplots(factor_num // 3 + 1, 3, dpi=400)
            for i in range(1, (3 - factor_num % 3) + 1):
                fig.delaxes(ax[factor_num // 3, i])

        ax = ax.flatten()
        for idx, col in enumerate(factor_index):
            lambda_k_column = self.lambda_k_array_mat[:, col]
            ax[idx].plot(self.timestamp_array, lambda_k_column, color='b')
            ax[idx].set_xlabel('time')
            ax[idx].set_xticks(self.timestamp_array)
            ax[idx].set_xticklabels([])
            ax[idx].set_ylabel(fr'$\lambda_{{{col + 1}}}(t)$')
            ax[idx].set_title(f'Topic {col + 1}')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    ibhp_ins = IBHP(n_sample=200)
    ibhp_ins.generate_data()
    ibhp_ins.plot_intensity_function()
    ibhp_ins.plot_each_factor_intensity(factor_num=9)
    print(ibhp_ins.text)
