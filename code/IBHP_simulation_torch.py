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
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
from matplotlib import rcParams


# noinspection SpellCheckingInspection,DuplicatedCode
class IBHPTorch:
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Times New Roman'

    def __init__(self,
                 doc_length: int,
                 word_num: int,
                 sum_kernel_num: int,
                 lambda0: torch.Tensor,
                 beta: torch.Tensor,
                 tau: torch.Tensor,
                 n_sample=100,
                 random_seed=None):
        self.random_seed = random_seed
        self.n_sample = n_sample
        self.lambda_k_tensor_mat = None
        self.sum_kernel_num = sum_kernel_num  # the number of base kernels
        self.D = doc_length  # length of each document
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
    def base_kernel(delta, beta, tau):
        base_kernel_l = beta * torch.exp(-delta / tau)
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
            delta_t_tensor.unsqueeze_(1)
            base_kernel_mat = self.base_kernel(delta_t_tensor, self.beta, self.tau)  # t_i for each row
            kappa_history = torch.einsum('lk,tl->tk', self.w, base_kernel_mat) * self.c
            kappa_history_count = torch.count_nonzero(self.c, dim=1).reshape(-1, 1)
            self.lambda_k_tensor = torch.sum(kappa_history / kappa_history_count, dim=0)

    def generate_timestamp_by_thinning(self, n):
        c_n_nonzero_index = torch.argwhere(self.c[-1] != 0)[:, 0]
        flag = False
        if n == 1:
            old_timestamp = torch.tensor(0.)
            old_intensity = self.lambda0
        else:
            old_timestamp = self.timestamp_tensor[-1]
            old_intensity = self.lambda_tn_tensor[-1]
        while not flag:
            if n == 1:
                # lambda star means conditional intensity function
                intensity_upper_bound = old_intensity + torch.sum(self.w[:, c_n_nonzero_index].T @ self.beta)
                t_n_candidate = old_timestamp + dist.Exponential(intensity_upper_bound).sample()
                candidate_timestamp_tensor = torch.tensor([t_n_candidate])
                old_timestamp = t_n_candidate
            else:
                intensity_upper_bound = old_intensity + torch.sum(self.w[:, c_n_nonzero_index].T @ self.beta)
                t_n_candidate = old_timestamp + dist.Exponential(intensity_upper_bound).sample()
                candidate_timestamp_tensor = self.timestamp_tensor.clone()
                candidate_timestamp_tensor = torch.hstack([candidate_timestamp_tensor, t_n_candidate])
                old_timestamp = t_n_candidate
            # compute lambda star tn
            delta_tn = candidate_timestamp_tensor[-1] - candidate_timestamp_tensor
            delta_tn.unsqueeze_(1)
            base_kernel_mat = self.base_kernel(delta_tn, self.beta, self.tau)
            kappa_t = torch.einsum('lk,tl->tk', self.w[:, c_n_nonzero_index],
                                   base_kernel_mat) * self.c[:delta_tn.shape[0], c_n_nonzero_index]
            kappa_t_count = torch.count_nonzero(self.c[:delta_tn.shape[0]], dim=1).reshape(-1, 1)
            lambda_star_tn = torch.sum(kappa_t / kappa_t_count) + self.lambda0
            old_intensity = lambda_star_tn
            u = dist.Uniform(0, intensity_upper_bound).sample()
            if u <= lambda_star_tn:
                if n == 1:
                    self.timestamp_tensor = torch.tensor([t_n_candidate])
                else:
                    self.timestamp_tensor = torch.hstack([self.timestamp_tensor, t_n_candidate])
                flag = True

    def generate_first_event(self):
        while self.K == 0:
            self.K = dist.Poisson(self.lambda0).sample()
        self.factor_num_tensor = self.K.unsqueeze(0)
        self.K = self.K.int()
        self.c = torch.ones((1, self.K))

        self.w = dist.Dirichlet(self.w_0).sample((self.K, )).T
        self.v = dist.Dirichlet(self.v_0).sample((self.K, )).T

        # sample t_1
        self.generate_timestamp_by_thinning(1)
        multi_dist_prob = torch.einsum(
            'ij->i', self.v[:, torch.argwhere(self.c[-1] != 0)[:, 0]]) / torch.count_nonzero(self.c[0])
        self.text = dist.Multinomial(self.D, multi_dist_prob).sample()

        # compute lambda_1
        self.calculate_lambda_k(1)
        c_n = torch.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_tensor = torch.sum(self.lambda_k_tensor[c_n], dim=0, keepdim=True) + self.lambda0

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

            w_new = dist.Dirichlet(self.w_0).sample((k_plus, )).T
            self.w = torch.hstack([self.w, w_new])
            v_new = dist.Dirichlet(self.v_0).sample((k_plus, )).T
            self.v = torch.hstack([self.v, v_new])
        else:
            self.c = torch.vstack([self.c, c_old])

        # sample t_n
        self.generate_timestamp_by_thinning(n)

        # sample T_n
        multi_dist_prob = torch.einsum(
            'ij->i', self.v[:, torch.argwhere(self.c[-1] != 0)[:, 0]]) / torch.count_nonzero(self.c[n - 1])
        text_n = dist.Multinomial(self.D, multi_dist_prob).sample()
        self.text = torch.vstack([self.text, text_n])

        # compute lambda_k
        self.calculate_lambda_k(n)
        c_n = torch.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_tensor = torch.hstack(
            [self.lambda_tn_tensor, torch.sum(self.lambda_k_tensor[c_n]) + self.lambda0])

    def generate_data(self, save_result=False, save_path: str = None):
        """
        save simulation data

        Args:
            save_result (bool, optional): _description_. if save_result is True, save simulation data. Defaults to False.
            save_path (str, optional): _description_. Path to save simulation data. Defaults to None.
        """
        logging.info('Begin generating simulation data')
        for i in torch.arange(1, self.n_sample + 1):
            if i == 1:
                self.generate_first_event()
            else:
                self.generate_following_event(i)
        logging.info(f'timestamp: \n{self.timestamp_tensor}')
        logging.info(f'text: \n{self.text}')
        if save_result and save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.w, f'{save_path}/w.pt')
            torch.save(self.v, f'{save_path}/v.pt')
            torch.save(self.c, f'{save_path}/c.pt')
            torch.save(self.timestamp_tensor, f'{save_path}/timestamp.pt')
            torch.save(self.text, f'{save_path}/text.pt')
            torch.save(self.lambda_tn_tensor, f'{save_path}/lambda_tn.pt')
            torch.save(self.factor_num_tensor, f'{save_path}/factor_num.pt')
            logging.info(f'Data saved to {save_path}')
        logging.info('End generating simulation data')

    def plot_intensity_function(self):
        fig, ax = plt.subplots(figsize=(26, 5), dpi=400)
        ax.scatter(self.timestamp_tensor, self.lambda_tn_tensor, color='red', marker='*', s=50)
        ax.plot(self.timestamp_tensor, self.lambda_tn_tensor, color='blue')
        ax.set_xlabel('event')
        ax.set_ylabel(r'$\lambda(t_n)$')
        ax.set_xticks(self.timestamp_tensor)
        ax.set_xticklabels([])
        if not os.path.exists('./img_test'):
            os.makedirs('./img_test')
        fig.savefig('./img_test/simulation_data_intensity.png')
        logging.info('Intensity function plot saved to ./img_test/simulation_data_intensity.png')
        plt.close('all')

    def plot_simulation_c_matrix(self):
        fig, ax = plt.subplots(dpi=400)
        ms = ax.matshow(self.c, cmap='YlGnBu')
        ax.set_ylabel('event')
        ax.set_xlabel('factors')
        fig.colorbar(ms, ax=ax)
        fig.tight_layout()
        if not os.path.exists('./img_test'):
            os.makedirs('./img_test')
        fig.savefig('./img_test/simulation_data_c_matrix.png')
        logging.info('C matrix plot saved to ./img_test/simulation_data_c_matrix.png')
        plt.close('all')


if __name__ == "__main__":
    ibhp_ins = IBHPTorch(
        n_sample=1000,
        random_seed=10,
        doc_length=20,
        word_num=1000,
        sum_kernel_num=3,
        lambda0=torch.tensor(2.),
        beta=torch.tensor([1., 2., 3.]),
        tau=torch.tensor([.3, .2, .1]),
    )
    ibhp_ins.generate_data(save_result=False, save_path='./model_result/simulation_data')
    ibhp_ins.plot_intensity_function()
    ibhp_ins.plot_simulation_c_matrix()
