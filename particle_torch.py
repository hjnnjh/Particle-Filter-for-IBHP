#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
@File    :   particle_torch.py
@Time    :   2022/4/2 17:24
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.distributions as dist
from functorch import vmap
from torch.nn.functional import softmax

from IBHP_simulation import IBHP

# ------------------------------ global vars ------------------------------
# when sample hyperparameter, allocate 3 gpu to compute sample likelihood separately
DEVICE0 = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
DEVICE1 = torch.device('cuda:1')
DEVICE2 = torch.device('cuda:2')
TENSOR = torch.Tensor


# noinspection SpellCheckingInspection,PyTypeChecker,DuplicatedCode
class Particle(IBHP):
    """
        This class implements all the steps of single particle sampling, hyperparameter updating and particle weight
        calculation.
    """
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    def __init__(self, word_corpus: TENSOR, timestamp_tensor: TENSOR, text_tensor: TENSOR, particle_idx: int,
                 sum_kernel_num: int = 3,
                 simulation_w: TENSOR = None, simulation_v: TENSOR = None, fix_w_v: bool = False,
                 random_seed: int = None, device: torch.device = DEVICE0):
        """
        :param word_corpus: all unique words
        :param timestamp_tensor: timestamp vector of real events
        :param text_tensor: text vector of real events
        :param sum_kernel_num: number of sum kernels
        :param simulation_w: w vector of simulation data
        :param simulation_v: v vector of simulation data
        :param fix_w_v: whether fix particle status
        """
        super(Particle, self).__init__()
        self.device = device
        self.random_seed = random_seed
        self.sum_kernel_num = torch.tensor(sum_kernel_num).to(self.device)
        self.text_tensor = text_tensor.to(self.device)
        self.timestamp_tensor = timestamp_tensor.to(self.device)
        self.word_corpus = word_corpus.to(self.device)
        self.word_num = word_corpus.shape[0]
        self.log_particle_weight = None
        self.lambda_k_tensor = None
        self.lambda_k_tensor_mat = None
        self.lambda_tn_tensor = None
        self.fix_w_v = fix_w_v
        self.particle_idx = particle_idx
        if self.fix_w_v:
            self.simulation_v = simulation_v.to(self.device)
            self.simulation_w = simulation_w.to(self.device)
            assert self.simulation_w.shape[1] == self.simulation_v.shape[1]
            self.simulation_factor_num = self.simulation_w.shape[1]
        if self.random_seed:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

    @staticmethod
    def exp_kernel(delta_t: TENSOR, beta: TENSOR, tau: TENSOR):
        """
        exp kernel
        :param delta_t:
        :param beta:
        :param tau:
        :return:
        """
        return beta * torch.exp(-delta_t / tau)

    def reset_particle_index(self, new_index: int):
        self.particle_idx = new_index

    def calculate_lambda_k(self, n):
        """
        compute lambda_k tensor for each event
        :param n:
        :return:
        """
        if n == 1:
            self.lambda_k_tensor = self.w.T @ self.beta
        else:
            delta_t_tensor = self.timestamp_tensor[n - 1] - self.timestamp_tensor[: n]
            exp_kernel_vfunc = vmap(self.exp_kernel, in_dims=(0, None, None))
            exp_kernel_mat = exp_kernel_vfunc(delta_t_tensor, self.beta, self.tau)
            kappa_history = torch.einsum('lk,tl->tk', self.w, exp_kernel_mat) * self.c
            kappa_history_count = torch.count_nonzero(self.c, dim=1).reshape(-1, 1)
            self.lambda_k_tensor = torch.sum(kappa_history / kappa_history_count, dim=0)

    def collect_factor_intensity(self, n):
        """
        Collect intensity of all factors generated during the simulation
        :param n: event
        :return:
        """
        if n == 1:
            self.lambda_k_tensor_mat = self.lambda_k_tensor.reshape(1, -1)
        else:
            zero_num = self.lambda_k_tensor.shape[0] - self.lambda_k_tensor_mat[-1].shape[0]
            if zero_num:
                self.lambda_k_tensor_mat = torch.hstack(
                    (self.lambda_k_tensor_mat,
                     torch.zeros((self.lambda_k_tensor_mat.shape[0], zero_num)).to(self.device)))
            self.lambda_k_tensor_mat = torch.vstack((self.lambda_k_tensor_mat, self.lambda_k_tensor))

    def sample_particle_first_event_status(self, lambda0: TENSOR, beta: TENSOR, tau: TENSOR):
        """
        generate the particle state corresponding to the first event
        :param tau:
        :param beta:
        :param lambda0:
        :return:
        """
        self.lambda0 = lambda0.to(self.device)
        self.beta = beta.to(self.device)
        self.tau = tau.to(self.device)
        self.w_0 = torch.tensor([1 / self.sum_kernel_num] * self.sum_kernel_num).to(self.device)
        self.v_0 = torch.tensor([1 / self.word_num] * self.word_num).to(self.device)

        # generate init K
        self.K = torch.poisson(self.lambda0)
        self.K = self.K.int()
        while self.K == 0:
            self.K = torch.poisson(self.lambda0)
            self.K = self.K.int()
        self.c = torch.ones((1, self.K)).to(self.device)

        # generate w and v
        if self.fix_w_v:  # fix w and v
            if self.simulation_factor_num >= self.K:
                self.w = self.simulation_w[:, : self.K]
                self.v = self.simulation_v[:, : self.K]
            else:
                w = dist.Dirichlet(self.w_0).sample((self.K - self.simulation_factor_num,)).T
                self.w = torch.hstack((self.simulation_w[:, : self.K], w))
                v = dist.Dirichlet(self.v_0).sample((self.K - self.simulation_factor_num,)).T
                self.v = torch.hstack((self.simulation_v[:, : self.K], v))
        else:
            self.w = dist.Dirichlet(self.w_0).sample((self.K,)).T
            self.v = dist.Dirichlet(self.v_0).sample((self.K,)).T

        # compute lambda_k_1
        self.calculate_lambda_k(1)
        c_1 = torch.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_tensor = torch.sum(self.lambda_k_tensor[c_1].reshape(1, -1), dim=1)
        logging.info(f'[event 1, particle {self.particle_idx}] factor occurence(c) shape: {self.c.shape}')
        # self.collect_factor_intensity(1)
        # logging.info(f'[event 1, particle {particle_idx + 1}] lambda k shape: {self.lambda_k_tensor_mat.shape}\n')

    def sample_particle_following_event_status(self, n: int):
        """
        generate the particle state corresponding to the following event
        :param n:
        :return:
        """
        p = self.lambda_k_tensor / ((self.lambda0 / self.K) + self.lambda_k_tensor)
        c_old = torch.bernoulli(p)
        k_plus = torch.poisson(self.lambda0 / (self.lambda0 + torch.sum(self.lambda_k_tensor)))
        k_plus = k_plus.int()
        # When K+ is equal to 0, check whether c_old is all 0, and if it is all 0, regenerate c_old
        if not k_plus:
            while torch.all(c_old == 0):
                c_old = torch.bernoulli(p)
        # update K
        self.K += k_plus
        if k_plus:
            c_new = torch.ones(k_plus).to(self.device)
            c = torch.hstack((c_old, c_new))
            self.c = torch.hstack(
                (self.c, torch.zeros((self.c.shape[0], k_plus)).to(self.device)))  # fill existing c matrix
            self.c = torch.vstack((self.c, c))

            # fix w and v
            if self.fix_w_v:
                if self.simulation_factor_num >= self.K:
                    self.w = self.simulation_w[:, : self.K]
                    self.v = self.simulation_v[:, : self.K]
                else:
                    w = dist.Dirichlet(self.w_0).sample((self.K - self.simulation_factor_num,)).T
                    self.w = torch.hstack((self.simulation_w[:, : self.K], w))
                    v = dist.Dirichlet(self.v_0).sample((self.K - self.simulation_factor_num,)).T
                    self.v = torch.hstack((self.simulation_v[:, : self.K], v))
            else:
                w_new = dist.Dirichlet(self.w_0).sample((k_plus,)).T
                self.w = torch.hstack((self.w, w_new))
                v_new = dist.Dirichlet(self.v_0).sample((k_plus,)).T
                self.v = torch.hstack((self.v, v_new))
        else:
            self.c = torch.vstack((self.c, c_old))

        # compute lambda_tn_k, lambda_tn
        self.calculate_lambda_k(n)
        c_n = torch.argwhere(self.c[-1] != 0)[:, 0]
        self.lambda_tn_tensor = torch.hstack((self.lambda_tn_tensor, torch.sum(self.lambda_k_tensor[c_n])))
        logging.info(f'[event {n}, particle {self.particle_idx}] factor occurence(c) shape: {self.c.shape}')
        # self.collect_factor_intensity(n)
        # logging.info(f'[event {n}, particle {particle_idx + 1}] lambda k shape: {self.lambda_k_tensor_mat.shape}\n')

    def log_hawkes_likelihood(self, n, allocated_device: torch.device,
                              lambda0: TENSOR, beta: TENSOR, tau: TENSOR):
        """
        log hawkes likelihood for IBHP
        :param allocated_device:
        :param n: event sequence index
        :param lambda0: like tensor(2.)
        :param beta: like tensor([1., 2., 3.])
        :param tau: like tensor([.1, .2, .3])
        :return:
        """
        # copy particle tensor to allocated gpu device
        lambda0 = lambda0.to(allocated_device)
        beta = beta.to(allocated_device)
        tau = tau.to(allocated_device)
        timestamp_tensor = self.timestamp_tensor.to(allocated_device)
        w = self.w.to(allocated_device)
        c = self.c.to(allocated_device)

        tau_unsqueezed = tau.unsqueeze(0)
        log_prod_term = torch.tensor(0.).to(allocated_device)
        sum_term = torch.tensor(0.).to(allocated_device)
        for i in torch.arange(1, n + 1):
            if i == 1:
                # just compute prod term when i=1
                prod_delta_ti_tj = timestamp_tensor[i - 1] - timestamp_tensor[: i]
                prod_delta_ti_tj.unsqueeze_(1)
                prod_exp_term = torch.exp(- prod_delta_ti_tj / tau)
                prod_exp_term.mul_(beta)
                prod_exp_term_einsum = torch.einsum('lk,tl->tk', w, prod_exp_term) * c[: i]
                kappa_j_count_prod = torch.count_nonzero(c[: i], dim=1).reshape(-1, 1)
                c_1 = torch.argwhere(c[i - 1] != 0)[:, 0]
                log_sum_1_prod = torch.sum(prod_exp_term_einsum[:, c_1] / kappa_j_count_prod)
                log_sum_1_prod.log_()
                log_prod_term = log_prod_term + log_sum_1_prod
            else:
                # integral term
                integral_delta_ti_1_tj = timestamp_tensor[i - 2] - timestamp_tensor[: i - 1]
                integral_delta_ti_1_tj.unsqueeze_(1)
                exp_ti_1_tj = torch.exp(- integral_delta_ti_1_tj / tau_unsqueezed)
                integral_delta_ti_tj = timestamp_tensor[i - 1] - timestamp_tensor[: i - 1]
                integral_delta_ti_tj.unsqueeze_(1)
                exp_ti_tj = torch.exp(- integral_delta_ti_tj / tau_unsqueezed)
                sum_exp_term = exp_ti_tj - exp_ti_1_tj
                sum_exp_term.mul_(beta)
                sum_exp_term.mul_(tau)
                sum_exp_term_einsum = torch.einsum('lk,tl->tk', w, sum_exp_term) * c[: i - 1]
                kappa_j_count_sum = torch.count_nonzero(c[: i - 1], dim=1).reshape(-1, 1)
                c_i = torch.argwhere(c[i - 1] != 0)[:, 0]  # shared
                sum_j_integral = torch.sum(sum_exp_term_einsum[:, c_i] / kappa_j_count_sum)
                sum_term = sum_term + sum_j_integral

                # prod term
                prod_delta_ti_tj = timestamp_tensor[i - 1] - timestamp_tensor[: i]
                prod_delta_ti_tj.unsqueeze_(1)
                prod_exp_term = torch.exp(- prod_delta_ti_tj / tau)
                prod_exp_term.mul_(beta)
                prod_exp_term_einsum = torch.einsum('lk,tl->tk', w, prod_exp_term) * c[: i]
                kappa_j_count_prod = torch.count_nonzero(c[: i], dim=1).reshape(-1, 1)
                log_sum_j_prod = torch.sum(prod_exp_term_einsum[:, c_i] / kappa_j_count_prod)
                log_sum_j_prod.log_()
                log_prod_term = log_prod_term + log_sum_j_prod

        if n == 1:
            log_integral_term = - lambda0 * timestamp_tensor[0]
        else:
            log_integral_term = - lambda0 * timestamp_tensor[0] + sum_term

        # log hawkes likelihood
        log_hawkes_likelihood = log_integral_term + log_prod_term
        return log_hawkes_likelihood

    def sample_hyperparameter(self, n: int,
                              allocated_device: torch.device,
                              alpha_lambda0: TENSOR,
                              alpha_beta: TENSOR,
                              alpha_tau: TENSOR,
                              random_num: int = 5000):
        """
        sample multiple groups of hyperparameter and compute prior and log hawkes likelihood
        :param allocated_device: allocated gpu device
        :param alpha_tau: concentration parameter for Gamma dist
        :param alpha_beta: concentration parameter for Gamma dist
        :param alpha_lambda0: concentration parameter for Gamma dist
        :param random_num:
        :param n:
        :return:
        """
        lambda0_gamma = dist.gamma.Gamma(alpha_lambda0, torch.tensor(1.))
        beta_gamma = dist.gamma.Gamma(alpha_beta, torch.tensor(1.))
        tau_gamma = dist.gamma.Gamma(alpha_tau, torch.tensor(1.))

        # draw samples from Gamma dist
        lambda0_candi_tensor = lambda0_gamma.sample((random_num,)).to(allocated_device)
        beta_candi_tensor_mat = beta_gamma.sample((random_num, self.sum_kernel_num)).to(allocated_device)
        tau_candi_tensor_mat = tau_gamma.sample((random_num, self.sum_kernel_num)).to(allocated_device)

        # log prior prob
        lambda0_candi_prior_log = lambda0_gamma.log_prob(lambda0_candi_tensor).to(allocated_device)
        beta_candi_prior_log = beta_gamma.log_prob(beta_candi_tensor_mat).to(allocated_device)
        tau_candi_prior_log = tau_gamma.log_prob(tau_candi_tensor_mat).to(allocated_device)

        # log hawkes likelihood for each set of samples, so fast now, wuhu~~~~~
        hawkes_likelihood_vfunc = vmap(partial(self.log_hawkes_likelihood, n, allocated_device), in_dims=(0, 0, 0))
        log_hawkes_likelihood_tensor = hawkes_likelihood_vfunc(lambda0_candi_tensor, beta_candi_tensor_mat,
                                                               tau_candi_tensor_mat)
        return lambda0_candi_tensor, lambda0_candi_prior_log, beta_candi_tensor_mat, beta_candi_prior_log, \
               tau_candi_tensor_mat, tau_candi_prior_log, log_hawkes_likelihood_tensor

        # log_weight_tensor = lambda0_candi_prior_log + torch.sum(beta_candi_prior_log, dim=1) + \
        #                     torch.sum(tau_candi_prior_log, dim=1) + log_hawkes_likelihood_tensor
        # weight_tensor = softmax(log_weight_tensor, 0)
        #
        # # fetch result and update hyperparameter
        # self.lambda0 = weight_tensor @ lambda0_candi_tensor
        # self.beta = weight_tensor @ beta_candi_tensor_mat
        # self.tau = weight_tensor @ tau_candi_tensor_mat

    def update_hyperparameter(self, n,
                              to_device: torch.device,
                              alpha_lambda0: TENSOR,
                              alpha_beta: TENSOR,
                              alpha_tau: TENSOR,
                              allocated_dict: Dict[torch.device, int]):
        """
        summarize the sampling results and calculate the weight of each group of samples
        :param allocated_dict:
        :param alpha_tau:
        :param alpha_beta:
        :param alpha_lambda0:
        :param n: event
        :param to_device: which gpu device should output be put
        :return:
        """
        total_lambda0_candi_tensor, total_lambda0_candi_prior_log, total_beta_candi_tensor_mat, \
        total_beta_candi_prior_log, total_tau_candi_tensor_mat, total_tau_candi_prior_log, \
        total_log_hawkes_likelihood_tensor = None, None, None, None, None, None, None
        for idx, (device, num) in enumerate(allocated_dict.items()):
            lambda0_candi_tensor, lambda0_candi_prior_log, beta_candi_tensor_mat, beta_candi_prior_log, \
            tau_candi_tensor_mat, tau_candi_prior_log, log_hawkes_likelihood_tensor = self.sample_hyperparameter(
                n=n,
                allocated_device=device,
                alpha_lambda0=alpha_lambda0,
                alpha_beta=alpha_beta,
                alpha_tau=alpha_tau,
                random_num=num
            )
            lambda0_candi_tensor = lambda0_candi_tensor.to(to_device)
            lambda0_candi_prior_log = lambda0_candi_prior_log.to(to_device)
            beta_candi_tensor_mat = beta_candi_tensor_mat.to(to_device)
            beta_candi_prior_log = beta_candi_prior_log.to(to_device)
            tau_candi_tensor_mat = tau_candi_tensor_mat.to(to_device)
            tau_candi_prior_log = tau_candi_prior_log.to(to_device)
            log_hawkes_likelihood_tensor = log_hawkes_likelihood_tensor.to(to_device)
            if idx == 0:
                total_lambda0_candi_tensor = lambda0_candi_tensor
                total_lambda0_candi_prior_log = lambda0_candi_prior_log
                total_beta_candi_tensor_mat = beta_candi_tensor_mat
                total_beta_candi_prior_log = beta_candi_prior_log
                total_tau_candi_tensor_mat = tau_candi_tensor_mat
                total_tau_candi_prior_log = tau_candi_prior_log
                total_log_hawkes_likelihood_tensor = log_hawkes_likelihood_tensor
            else:
                total_lambda0_candi_tensor = torch.hstack((total_lambda0_candi_tensor, lambda0_candi_tensor))
                total_lambda0_candi_prior_log = torch.hstack((total_lambda0_candi_prior_log, lambda0_candi_prior_log))
                total_beta_candi_tensor_mat = torch.vstack((total_beta_candi_tensor_mat, beta_candi_tensor_mat))
                total_beta_candi_prior_log = torch.vstack((total_beta_candi_prior_log, beta_candi_prior_log))
                total_tau_candi_tensor_mat = torch.vstack((total_tau_candi_tensor_mat, tau_candi_tensor_mat))
                total_tau_candi_prior_log = torch.vstack((total_tau_candi_prior_log, tau_candi_prior_log))
                total_log_hawkes_likelihood_tensor = torch.hstack((total_log_hawkes_likelihood_tensor,
                                                                   log_hawkes_likelihood_tensor))

        log_weight_tensor = total_lambda0_candi_prior_log + torch.sum(total_beta_candi_prior_log, dim=1) + \
                            torch.sum(total_tau_candi_prior_log, dim=1) + total_log_hawkes_likelihood_tensor
        weight_tensor = softmax(log_weight_tensor, 0)
        # fetch result and update hyperparameter
        self.lambda0 = weight_tensor @ total_lambda0_candi_tensor
        self.beta = weight_tensor @ total_beta_candi_tensor_mat
        self.tau = weight_tensor @ total_tau_candi_tensor_mat

    def update_log_particle_weight(self, n, old_particle_weight: TENSOR):
        """
        calculate and update log particle weight
        :param n:
        :param old_particle_weight:
        :return:
        """
        tau_unsqueezed = self.tau.unsqueeze(0)
        c_n = torch.argwhere(self.c[n - 1] != 0)[:, 0]
        # likelihood of timestamp
        if n == 1:
            log_likelihood_timestamp = torch.log(self.lambda0 * self.timestamp_tensor[0])
        else:
            integral_delta_tn_1_ti = self.timestamp_tensor[n - 2] - self.timestamp_tensor[: n]
            integral_delta_tn_1_ti.unsqueeze_(1)
            exp_tn_1_ti = torch.exp(- integral_delta_tn_1_ti / tau_unsqueezed)
            integral_delta_tn_ti = self.timestamp_tensor[n - 1] - self.timestamp_tensor[: n]
            integral_delta_tn_ti.unsqueeze_(1)
            exp_tn_ti = torch.exp(- integral_delta_tn_ti / tau_unsqueezed)
            exp_term = exp_tn_1_ti - exp_tn_ti
            exp_term.mul_(self.beta)
            exp_term.mul_(self.tau)
            exp_term_einsum = torch.einsum('lk,tl->tk', self.w, exp_term) * self.c[: n]
            kappa_i_count = torch.count_nonzero(self.c[: n], dim=1).reshape(-1, 1)
            log_likelihood_timestamp = torch.log(torch.sum(exp_term_einsum[:, c_n] / kappa_i_count))
        if torch.isinf(log_likelihood_timestamp):  # deal with `inf`
            log_likelihood_timestamp = torch.tensor(0.).to(self.device)
        # likelihood of text
        vn_avg = torch.einsum('ij->i', self.v[:, c_n]) / torch.count_nonzero(self.c[n - 1])
        text_n = self.text_tensor[n - 1]
        vn_avg.log_()
        log_likelihood_text = torch.sum(text_n * vn_avg)

        self.log_particle_weight = torch.log(old_particle_weight) + log_likelihood_timestamp + log_likelihood_text
