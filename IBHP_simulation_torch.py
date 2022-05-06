#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   IBHP_simulation_torch.py
@Time    :   2022/4/24 18:53
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from matplotlib import rcParams
import numpy as np
from functorch import vmap


# noinspection SpellCheckingInspection,DuplicatedCode
class IBHPTorch:
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Times New Roman'

    def __init__(self, doc_len: int, word_num: int, sum_kernel_num,
                 lambda0: float, beta: torch.Tensor, tau: torch.Tensor,
                 n_sample=100, random_seed=None):
        self.random_seed = random_seed
        self.n_sample = n_sample
        self.lambda_k_tensor_mat = None
        self.sum_kernel_num = sum_kernel_num  # the number of base kernels
        self.D = doc_len  # length of each document
        self.word_num = word_num  # The total number of words in the dictionary
        self.w_0 = torch.tensor([1 / self.sum_kernel_num] * self.sum_kernel_num)  # The topic distribution (dirichlet)
        # parameter of the document
        self.w = None  # The weight of the base kernel
        self.v_0 = torch.tensor([1 / self.word_num] * self.word_num)  # topic word distribution (dirichlet) parameter
        self.v = None  # word distribution
        self.lambda_k_tensor = None
        self.beta = beta  # base kernel initial parameters
        self.tau = tau  # base kernel initial parameters
        self.c = None  # topic vector
        self.timestamp_tensor = None  # Timestamp for each sample
        self.text = None  # sample text vector
        self.K = 0  # Number of topics
        self.factor_num_tensor = None
        self.lambda0 = lambda0  # base rate
        self.lambda_tn_tensor = None  # rate array
        if self.random_seed:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

    @staticmethod
    def base_kernel_l(delta, beta, tao):
        base_kernel_l = beta * torch.exp(- delta / tao)
        return base_kernel_l

    def calculate_lambda_k(self, n):
        """
        calculate lambda_k array for each event
        :param n:
        :return:
        """
        if n == 1:
            self.lambda_k_tensor = self.w.T @ self.beta
        elif n >= 2:
            delta_t_tensor = self.timestamp_tensor[-1] - self.timestamp_tensor

            kernel_vfunc = vmap(self.base_kernel_l, in_dims=(0, None, None))
            base_kernel_mat = kernel_vfunc(delta_t_tensor, self.beta, self.tau)  # t_i for each row
            kappa_history = torch.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c
            kappa_history_count = torch.count_nonzero(self.c, dim=1).reshape(-1, 1)
            self.lambda_k_tensor = torch.sum(kappa_history / kappa_history_count, dim=0)

    def generate_timestamp_by_thinning(self, n):
        flag = None
        while not flag:
            if n == 1:
                lambda_star = self.lambda0
                u = dist.Uniform(0., 1.).sample()
                candidate_next_timestamp = - (1 / lambda_star) * torch.log(u)
                candidate_timestamp_tensor = torch.tensor([candidate_next_timestamp])
            else:
                lambda_star = self.lambda_tn_tensor[-1]
                u = dist.Uniform(0., 1.).sample()
                time_interval = - (1 / lambda_star) * torch.log(u)
                candidate_next_timestamp = self.timestamp_tensor[-1] + time_interval
                candidate_timestamp_tensor = self.timestamp_tensor.clone()
                candidate_timestamp_tensor = torch.hstack([candidate_timestamp_tensor, candidate_next_timestamp])
            delta_t = candidate_timestamp_tensor[-1] - candidate_timestamp_tensor
            kernel_vfunc = vmap(self.base_kernel_l, in_dims=(0, None, None))
            base_kernel_mat = kernel_vfunc(delta_t, self.beta, self.tau)
            kappa_t = torch.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c[: delta_t.shape[0]]
            kappa_t_count = torch.count_nonzero(self.c[: delta_t.shape[0]], dim=1).reshape(-1, 1)
            lambda_k_tensor = torch.sum(kappa_t / kappa_t_count, dim=0)
            c_t = torch.argwhere(self.c[-1] != 0)[:, 0]
            lambda_t = torch.sum(lambda_k_tensor[c_t])
            # rejection test
            s = dist.Uniform(0., 1.).sample()
            if s <= lambda_t / lambda_star:
                if n == 1:
                    self.timestamp_tensor = torch.tensor([candidate_next_timestamp])
                else:
                    self.timestamp_tensor = torch.hstack([self.timestamp_tensor, candidate_next_timestamp])
                flag = "PASS"
            else:
                continue

    def generate_first_event(self):
        while self.K == 0:
            self.K = dist.Poisson(self.lambda0).sample()
        self.factor_num_tensor = self.K.unsqueeze(0)
        self.K = self.K.int()
        self.c = torch.ones((1, self.K))

        self.w = dist.Dirichlet(self.w_0).sample((self.K,)).T
        self.v = dist.Dirichlet(self.v_0).sample((self.K,)).T

        # sample t_1
        self.generate_timestamp_by_thinning(1)
        multi_dist_prob = torch.einsum('ij->i', self.v[:, torch.argwhere(self.c[-1] != 0)[:, 0]]) / \
                          torch.count_nonzero(self.c[0])
        self.text = dist.Multinomial(self.D, multi_dist_prob).sample()

        # compute lambda_1
        self.calculate_lambda_k(1)
        c_n = torch.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_tensor = torch.sum(self.lambda_k_tensor[c_n], dim=0, keepdim=True)

    def generate_following_event(self, n):
        p = self.lambda_k_tensor / ((self.lambda0 / self.K) + self.lambda_k_tensor)

        c_old = dist.Bernoulli(p).sample()
        k_plus = dist.Poisson(self.lambda0 / (self.lambda0 + torch.sum(self.lambda_k_tensor))).sample()
        if k_plus == 0:
            while torch.all(c_old == 0):
                c_old = dist.Bernoulli(p).sample()
        k_plus = k_plus.int()
        self.K += k_plus
        self.factor_num_tensor = torch.hstack([self.factor_num_tensor, self.K])
        if k_plus:
            c_new = torch.ones(k_plus)
            c = torch.hstack([c_old, c_new])
            self.c = torch.hstack([self.c, torch.zeros([self.c.shape[0], k_plus])])
            self.c = torch.vstack([self.c, c])

            w_new = dist.Dirichlet(self.w_0).sample((k_plus,)).T
            self.w = torch.hstack([self.w, w_new])
            v_new = dist.Dirichlet(self.v_0).sample((k_plus,)).T
            self.v = torch.hstack([self.v, v_new])
        else:
            self.c = torch.vstack([self.c, c_old])

        # sample t_n
        self.generate_timestamp_by_thinning(n)

        # sample T_n
        multi_dist_prob = torch.einsum('ij->i', self.v[:, torch.argwhere(self.c[-1] != 0)[:, 0]]) / \
                          torch.count_nonzero(self.c[n - 1])
        text_n = dist.Multinomial(self.D, multi_dist_prob).sample()
        self.text = torch.vstack([self.text, text_n])

        # compute lambda_k
        self.calculate_lambda_k(n)
        c_n = torch.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_tensor = torch.hstack([self.lambda_tn_tensor,
                                              torch.sum(self.lambda_k_tensor[c_n])])

    def generate_data(self):
        logging.info('Begin generating simulation data')
        for i in torch.arange(1, self.n_sample + 1):
            if i == 1:
                self.generate_first_event()
            else:
                self.generate_following_event(i)
        logging.info(f'timestamp: \n{self.timestamp_tensor}')
        logging.info(f'text: \n{self.text}')

    def plot_intensity_function(self):
        fig, ax = plt.subplots(dpi=500)
        ax.plot(self.timestamp_tensor, self.lambda_tn_tensor, color='tomato')
        ax.set_xlabel('event')
        ax.set_ylabel(r'$\lambda(t_n)$')
        ax.set_xticks(self.timestamp_tensor)
        ax.set_xticklabels([])
        plt.show()


if __name__ == "__main__":
    ibhp_ins = IBHPTorch(
        n_sample=250,
        random_seed=2,
        doc_len=20,
        word_num=1000,
        sum_kernel_num=3,
        lambda0=2.,
        beta=torch.tensor([1., 2., 3.]),
        tau=torch.tensor([.3, .2, .1]),
    )
    ibhp_ins.generate_data()
    ibhp_ins.plot_intensity_function()
