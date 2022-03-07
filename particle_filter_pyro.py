#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection GrazieInspection,SpellCheckingInspection
"""
@File    :   particle_filter_pyro.py
@Time    :   2021/12/15 4:10 PM
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None

In pyro.SMCFilter, model is used to generate simulation data, guide is used to generate particle status for each
timestamp, I got it.
"""

import logging
from functools import partial

import pyro
import pyro.distributions as dist
import torch
from torch import tensor

device = torch.device('cuda')


# noinspection SpellCheckingInspection,PyUnresolvedReferences,PyPep8Naming,DuplicatedCode
class IBHP_Model:
    """
    IBHP model for each particle
    """

    # configuration of log output
    # noinspection SpellCheckingInspection
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d--%H:%M:%S")
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.WARNING, datefmt="%Y-%m-%d--%H:%M:%S")

    def __init__(self, word_dict: tensor, n_sample: int, base_kernel_num: int = 3):
        """
        basic params of IBHP model
        :param word_dict:
        :param base_kernel_num:
        """
        self.n_sample = n_sample
        self.text_tensor = None
        self.timestamp_tensor = None
        self.base_kernel_num = base_kernel_num
        self.word_dict = word_dict
        self.word_num = word_dict.shape[0]
        self.lambda_0 = pyro.sample('lambda_0', lambda: tensor(2.).to(device))
        self.beta = pyro.sample('beta', lambda: tensor([2.0, 2.0, 2.0]).to(device))
        self.tau = pyro.sample('tau', lambda: tensor([0.2, 0.2, 0.2]).to(device))
        self.w_0 = pyro.param('w_0', tensor([1 / self.base_kernel_num] * self.base_kernel_num).to(device))
        self.v_0 = pyro.param('v_0', tensor([1 / self.word_num] * self.word_num).to(device))
        self.document_length = 20
        self.t = None
        self.s = None
        self.particle_state = {}

    # --------------------- utils ---------------------
    @staticmethod
    def base_kernel(beta, tau, delta):
        base_kernel = beta * torch.exp(- delta / tau)
        return base_kernel

    def calculate_lambda_k(self):
        if self.t == 1:
            self.particle_state['lambda_k'] = pyro.sample(f'lambda_k_{self.t}',
                                                          lambda: self.particle_state['w'] @ self.beta)
        else:
            delta_t_tensor = self.timestamp_tensor[self.t - 1] - self.timestamp_tensor[: self.t]
            base_kernel_func = partial(self.base_kernel, self.beta, self.tau)
            base_kernel_for_each_timestamp = list(map(base_kernel_func, delta_t_tensor))
            base_kernel_for_each_timestamp = torch.stack(base_kernel_for_each_timestamp)
            kappa_history = torch.einsum('kl,tl->tk', self.particle_state['w'], base_kernel_for_each_timestamp) * \
                            self.particle_state['c'][: delta_t_tensor.shape[0]]
            kappa_history_count = torch.count_nonzero(self.particle_state['c'][: delta_t_tensor.shape[0]],
                                                      dim=1).reshape(-1, 1)
            self.particle_state['lambda_k'] = pyro.sample(f'lambda_k_{self.t}',
                                                          lambda: torch.sum(kappa_history / kappa_history_count, dim=0))

    def generate_timestamp(self):
        if self.t == 1:
            lambda_star = self.lambda_0
            u = pyro.sample(f'u_{self.t}', dist.Uniform(0, 1)).to(device)
            self.s = - (1 / lambda_star) * torch.log(u)
            self.timestamp_tensor = pyro.sample(f'timestamp_{self.t}', lambda: tensor([self.s])).to(device)
        else:
            FLAG = 'failed'
            # update maximum intensity
            kappa_nonzero_index = torch.where(self.particle_state['c'][self.t - 1] != 0)[0]
            Y = torch.sum(self.particle_state['w'][kappa_nonzero_index, :] @ self.beta) / torch.count_nonzero(
                self.particle_state['c'][self.t - 1])
            lambda_star = self.particle_state['lambda_t'][-1] + Y + self.lambda_0
            while FLAG is 'failed':
                u = pyro.sample(f'u_{self.t}', dist.Uniform(0, 1)).to(device)
                self.s = self.s - (1 / lambda_star) * torch.log(u)
                if self.t > self.n_sample:
                    return 'exit'
                # rejection test, calculate lambda(s)
                delta_s = self.s - self.timestamp_tensor[self.timestamp_tensor <= self.s]
                base_kernel_func = partial(self.base_kernel, self.beta, self.tau)
                base_kernel_for_each_timestamp = list(map(base_kernel_func, delta_s))
                base_kernel_for_each_timestamp = torch.stack(base_kernel_for_each_timestamp)
                kappa_s = torch.einsum('kl,tl->tk', self.particle_state['w'], base_kernel_for_each_timestamp) * \
                          self.particle_state['c'][: delta_s.shape[0]]
                kappa_s_nonzero_index = torch.where(self.particle_state['c'][delta_s.shape[0] - 1] != 0)[0]
                kappa_s_count = torch.count_nonzero(self.particle_state['c'][: delta_s.shape[0]], dim=1).reshape(-1, 1)
                lambda_s_tensor = torch.sum(kappa_s / kappa_s_count, dim=0)
                lambda_s = torch.sum(lambda_s_tensor[kappa_s_nonzero_index]) + self.lambda_0

                d = pyro.sample(f'd_{self.t}', dist.Uniform(0, 1)).to(device)
                if d <= lambda_s / lambda_star:
                    self.timestamp_tensor = pyro.sample(f'timestamp_{self.t}',
                                                        lambda: torch.hstack((self.timestamp_tensor, self.s)))
                    FLAG = 'pass'
                else:
                    lambda_star = lambda_s

    # --------------------- steps for SMCFilter ---------------------
    def init(self):
        """
        generate initial states for each particle, t=1
        :return: None
        """
        self.t = 1
        self.particle_state['K'] = pyro.sample(f'K_{self.t}', dist.Poisson(self.lambda_0))

        # generate state for event 1
        while self.particle_state['K'] == 0:
            logging.warning(f'[event 1] topic number K is 0, regenerate')
            self.particle_state['K'] = pyro.sample(f'K_{self.t}', dist.Poisson(self.lambda_0))

        self.particle_state['c'] = pyro.sample(f'c_{self.t}',
                                               lambda: torch.ones((1, int(self.particle_state['K']))).to(device))
        with pyro.plate(f'multi_w_{self.t}', self.particle_state['K']):
            self.particle_state['w'] = pyro.sample(f'w_{self.t}', dist.Dirichlet(self.w_0))  # (K, L)
        with pyro.plate(f'multi_v_{self.t}', self.particle_state['K']):
            self.particle_state['v'] = pyro.sample(f'v_{self.t}', dist.Dirichlet(self.v_0))  # (K, word_num)

        self.particle_state['kappa'] = pyro.sample(f'kappa_{self.t}', lambda: self.particle_state['w'] @ self.beta)
        self.calculate_lambda_k()
        self.particle_state['lambda_t'] = pyro.sample(f'lambda_t_{self.t}',
                                                      lambda: tensor([torch.sum(self.particle_state['lambda_k'])]).to(
                                                          device))
        # generate timestamp
        self.generate_timestamp()

        multi_dist_prob = torch.einsum('ij->j',
                                       self.particle_state['v'][
                                       torch.where(self.particle_state['c'][self.t - 1] != 0)[0],
                                       :]) / torch.count_nonzero(self.particle_state['c'][self.t - 1])
        self.text_tensor = pyro.sample(f'text_{self.t}', dist.Multinomial(self.document_length, multi_dist_prob))

    def step(self):
        """
        generate the following states for each particle
        :return:
        """
        self.t += 1
        # the probability of generating c from the existing K topics
        self.particle_state['p'] = pyro.sample(f'p_{self.t}', lambda: self.particle_state['lambda_k'] /
                                                                      (self.lambda_0 / self.particle_state['K'] +
                                                                       self.particle_state['lambda_k']))
        self.particle_state['p'][torch.isnan(self.particle_state['p'])] = 0
        c_old = pyro.sample(f'c_old_{self.t}', dist.Bernoulli(self.particle_state['p']))
        K_plus = pyro.sample(f'K_plus_{self.t}', dist.Poisson(
            self.lambda_0 / (self.lambda_0 + torch.sum(self.particle_state['lambda_k']))))
        if K_plus == 0:
            while torch.all(c_old == 0):
                logging.warning(f'[event {self.t}] the occurrence of the topic is all 0, resampling')
                c_old = pyro.sample(f'c_old_{self.t}', dist.Bernoulli(self.particle_state['p']))
        # update K
        self.particle_state['K'] = pyro.sample(f'K_{self.t}', lambda: self.particle_state['K'] + K_plus)
        if K_plus:
            # update topic occurrence
            c_new = torch.ones(int(K_plus)).to(device)
            c_t = torch.hstack((c_old, c_new))
            self.particle_state['c'] = torch.hstack(
                (self.particle_state['c'],
                 torch.zeros(
                     (self.particle_state['c'].shape[0], int(K_plus))).to(
                     device)))  # Complete the existing c matrix with 0
            self.particle_state['c'] = pyro.sample(f'c_{self.t}',
                                                   lambda: torch.vstack((self.particle_state['c'], c_t)))

            # update w
            with pyro.plate(f'w_new_plate_{self.t}', K_plus):
                w_new = pyro.sample(f'w_new_{self.t}', dist.Dirichlet(self.w_0))
            self.particle_state['w'] = pyro.sample(f'w_{self.t}',
                                                   lambda: torch.vstack((self.particle_state['w'], w_new)))

            # update v
            with pyro.plate(f'v_new_plate_{self.t}', K_plus):
                v_new = pyro.sample(f'v_new_{self.t}', dist.Dirichlet(self.v_0))
            self.particle_state['v'] = pyro.sample(f'v_{self.t}',
                                                   lambda: torch.vstack((self.particle_state['v'], v_new)))

            # update kappa
            kappa_new = w_new @ self.beta
            self.particle_state['kappa'] = self.particle_state['w'][: c_old.shape[0], :] @ self.beta * c_old
            self.particle_state['kappa'] = pyro.sample(f'kappa_{self.t}',
                                                       lambda: torch.hstack((self.particle_state['kappa'], kappa_new)))
        else:  # no new topic appear
            self.particle_state['c'] = pyro.sample(f'c_{self.t}',
                                                   lambda: torch.vstack((self.particle_state['c'], c_old)))
            self.particle_state['kappa'] = pyro.sample(f'kappa_{self.t}',
                                                       lambda: self.particle_state['w'] @ self.beta * c_old)

        # sample t
        self.generate_timestamp()

        # sample text
        multi_dist_prob = torch.einsum('ij->j',
                                       self.particle_state['v'][
                                       torch.where(self.particle_state['c'][self.t - 1] != 0)[0],
                                       :]) / torch.count_nonzero(self.particle_state['c'][self.t - 1])
        self.text_tensor = pyro.sample(f'text_{self.t}', lambda: torch.vstack((self.text_tensor,
                                                                               pyro.sample(f'text_{self.t}',
                                                                                           dist.Multinomial(
                                                                                               self.document_length,
                                                                                               multi_dist_prob)))))

        # calculate lambda_k and lambda_t
        self.calculate_lambda_k()
        kappa_nonzero_index = torch.where(self.particle_state['kappa'] != 0)[0]
        self.particle_state['lambda_t'] = torch.hstack(
            (self.particle_state['lambda_t'], torch.sum(self.particle_state['lambda_k'][kappa_nonzero_index])))


# noinspection DuplicatedCode,PyUnresolvedReferences,PyPep8Naming
class IBHP_Particle(IBHP_Model):

    def __init__(self, n_sample, word_dict, timestamp_tensor, text_tensor, base_kernel_num=3):
        super(IBHP_Particle, self).__init__(word_dict=word_dict, n_sample=n_sample, base_kernel_num=base_kernel_num)
        self.timestamp_tensor = timestamp_tensor
        self.text_tensor = text_tensor

    # ------------------------- utils -------------------------
    def log_likelihood(self):
        pass

    def init(self):
        self.t = 1
        self.particle_state['K'] = pyro.sample(f'K_{self.t}', dist.Poisson(self.lambda_0))

        # generate state for event 1
        while self.particle_state['K'] == 0:
            logging.warning(f'[event 1] topic number K is 0, regenerate')
            self.particle_state['K'] = pyro.sample(f'K_{self.t}', dist.Poisson(self.lambda_0))

        self.particle_state['c'] = torch.ones((1, int(self.particle_state['K'])))
        with pyro.plate(f'multi_w_{self.t}', self.particle_state['K']):
            self.particle_state['w'] = pyro.sample(f'w_{self.t}', dist.Dirichlet(self.w_0))  # (K, L)
        with pyro.plate(f'multi_v_{self.t}', self.particle_state['K']):
            self.particle_state['v'] = pyro.sample(f'v_{self.t}', dist.Dirichlet(self.v_0))  # (K, word_num)

        self.particle_state['kappa'] = self.particle_state['w'] @ self.beta
        self.calculate_lambda_k()
        self.particle_state['lambda_t'] = tensor([torch.sum(self.particle_state['lambda_k'])])

    def step(self):
        self.t += 1
        # the probability of generating c from the existing K topics
        self.particle_state['p'] = pyro.param(f'p_{self.t}', lambda: self.particle_state['lambda_k'] /
                                                                     (self.lambda_0 / self.particle_state['K'] +
                                                                      self.particle_state['lambda_k']))
        self.particle_state['p'][torch.isnan(self.particle_state['p'])] = 0
        c_old = pyro.sample(f'c_old_{self.t}', dist.Bernoulli(self.particle_state['p']))
        print()
        K_plus = pyro.sample(f'K_plus_{self.t}', dist.Poisson(
            self.lambda_0 / (self.lambda_0 + torch.sum(self.particle_state['lambda_k']))))
        if K_plus == 0:
            while torch.all(c_old == 0):
                logging.warning(f'[event {self.t}] the occurrence of the topic is all 0, resampling')
                c_old = pyro.sample(f'c_old_{self.t}', dist.Bernoulli(self.particle_state['p']))
        # update K
        self.particle_state['K'] += K_plus
        if K_plus:
            # update topic occurrence
            c_new = torch.ones(int(K_plus))
            c_t = torch.hstack((c_old, c_new))
            self.particle_state['c'] = torch.hstack(
                (self.particle_state['c'],
                 torch.zeros(
                     (self.particle_state['c'].shape[0], int(K_plus)))))  # Complete the existing c matrix with 0
            self.particle_state['c'] = torch.vstack((self.particle_state['c'], c_t))

            # update w
            with pyro.plate(f'w_new_plate_{self.t}', K_plus):
                w_new = pyro.sample(f'w_new_{self.t}', dist.Dirichlet(self.w_0))
            self.particle_state['w'] = torch.vstack((self.particle_state['w'], w_new))

            # update v
            with pyro.plate(f'v_new_plate_{self.t}', K_plus):
                v_new = pyro.sample(f'v_new_{self.t}', dist.Dirichlet(self.v_0))
            self.particle_state['v'] = torch.vstack((self.particle_state['v'], v_new))

            # update kappa
            kappa_new = w_new @ self.beta
            self.particle_state['kappa'] = self.particle_state['w'][: c_old.shape[0], :] @ self.beta * c_old
            self.particle_state['kappa'] = torch.hstack((self.particle_state['kappa'], kappa_new))
        else:  # no new topic appear
            self.particle_state['c'] = torch.vstack((self.particle_state['c'], c_old))
            self.particle_state['kappa'] = self.particle_state['w'] @ self.beta * c_old

            # calculate lambda_k and lambda_t
            self.calculate_lambda_k()
            kappa_nonzero_index = torch.where(self.particle_state['kappa'] != 0)[0]
            self.particle_state['lambda_t'] = torch.hstack(
                (self.particle_state['lambda_t'], torch.sum(self.particle_state['lambda_k'][kappa_nonzero_index])))
