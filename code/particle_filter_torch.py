#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   particle_filter_torch.py
@Time    :   2022/4/14 17:52
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import argparse
import logging
import os
import shutil
from ast import arg
from copy import deepcopy
from datetime import datetime

import torch
import torch.distributions as dist
from torch.nn.functional import softmax

from IBHP_simulation_torch import IBHPTorch
from particle_torch import (DEVICE0, TENSOR, HyperparameterFixedParticle, Particle, StatesFixedParticle)


# noinspection PyShadowingNames,DuplicatedCode,PyUnresolvedReferences
class ParticleFilter:
    def __init__(self,
                 n_particle: int,
                 n_sample: int,
                 word_corpus: TENSOR,
                 sum_kernel_num: int,
                 lambda0: TENSOR,
                 beta: TENSOR,
                 tau: TENSOR,
                 alpha_lambda0: TENSOR,
                 alpha_beta: TENSOR,
                 alpha_tau: TENSOR,
                 random_num: int,
                 fix_w_v: bool = False,
                 ibhp_ins: IBHPTorch = None,
                 text_tensor: TENSOR = None,
                 timestamp_tensor: TENSOR = None,
                 states_fixed: bool = False,
                 hyperparameter_fixed: bool = False,
                 fix_beta: bool = False,
                 fix_tau: bool = False,
                 device: torch.device = DEVICE0,
                 chunk: bool = False):
        """
        Particle Fileter __init__

        Args:
            n_particle (int): _description_
            n_sample (int): _description_
            word_corpus (TENSOR): _description_
            sum_kernel_num (int): _description_
            lambda0 (TENSOR): _description_
            beta (TENSOR): _description_
            tau (TENSOR): _description_
            alpha_lambda0 (TENSOR): _description_
            alpha_beta (TENSOR): _description_
            alpha_tau (TENSOR): _description_
            random_num (int): _description_
            fix_w_v (bool, optional): _description_. Defaults to False.
            ibhp_ins (IBHPTorch, optional): _description_. Defaults to None.
            text_tensor (TENSOR, optional): _description_. Defaults to None.
            timestamp_tensor (TENSOR, optional): _description_. Defaults to None.
            states_fixed (bool, optional): _description_. Defaults to False.
            hyperparameter_fixed (bool, optional): _description_. Defaults to False.
            fix_beta (bool, optional): _description_. Defaults to False.
            fix_tau (bool, optional): _description_. Defaults to False.
            device (torch.device, optional): _description_. Defaults to DEVICE0.
            chunk (bool, optional): _description_. Defaults to False.
        """
        self.word_corpus = word_corpus
        self.random_num = random_num
        self.alpha_tau = alpha_tau
        self.alpha_beta = alpha_beta
        self.alpha_lambda0 = alpha_lambda0
        self.tau = tau
        self.beta = beta
        self.lambda0 = lambda0
        self.n_sample = n_sample
        self.n_particle = n_particle
        self.device = device
        self.text_tensor = text_tensor
        self.timestamp_tensor = timestamp_tensor
        self.ibhp_ins = ibhp_ins
        self.sum_kernel_num = sum_kernel_num
        self.states_fixed = states_fixed
        self.hyperparameter_fixed = hyperparameter_fixed
        self.fix_beta = fix_beta
        self.fix_tau = fix_tau
        if self.ibhp_ins:
            self.true_lambda_tn = self.ibhp_ins.lambda_tn_tensor
            self.timestamp_tensor = self.ibhp_ins.timestamp_tensor
            self.text_tensor = self.ibhp_ins.text
        # `self.text_tensor` and `self.timestamp_tensor` should not be `None`
        assert isinstance(self.timestamp_tensor, TENSOR)
        assert isinstance(self.text_tensor, TENSOR)
        if self.states_fixed:
            self.particle_list = [
                StatesFixedParticle(ibhp=self.ibhp_ins,
                                    particle_idx=i,
                                    word_corpus=self.word_corpus,
                                    lambda0=self.lambda0,
                                    beta=self.beta,
                                    tau=self.tau,
                                    sum_kernel_num=self.sum_kernel_num,
                                    device=self.device,
                                    chunk=chunk) for i in range(self.n_particle)
            ]
        elif self.hyperparameter_fixed:
            self.particle_list = [
                HyperparameterFixedParticle(word_corpus=self.word_corpus,
                                            particle_idx=i,
                                            ibhp=self.ibhp_ins,
                                            fix_w_v=fix_w_v,
                                            device=self.device) for i in range(self.n_particle)
            ]
        else:
            self.particle_list = [
                Particle(word_corpus=self.word_corpus,
                         timestamp_tensor=self.timestamp_tensor,
                         text_tensor=self.text_tensor,
                         particle_idx=i,
                         sum_kernel_num=self.sum_kernel_num,
                         device=self.device,
                         fix_w_v=fix_w_v,
                         ibhp=self.ibhp_ins,
                         chunk=chunk) for i in range(self.n_particle)
            ]
        self.particle_weight_tensor = torch.tensor([1 / self.n_particle
                                                    for i in torch.arange(self.n_particle)]).to(self.device)

    def _generate_status_for_states_fixed_particles(self, n: int):
        """generate status for states fixed particles

        Args:
            n (int): _description_
        """
        for particle in self.particle_list:
            particle.compute_lambda_tn(n)
            logging.info(f'[event {n}, states fixed particle {particle.particle_idx}] Updating hyper-parameter')
            particle.update_hyperparameter(n=n,
                                           alpha_lambda0=self.alpha_lambda0,
                                           alpha_beta=self.alpha_beta,
                                           alpha_tau=self.alpha_tau,
                                           fix_beta=self.fix_beta,
                                           fix_tau=self.fix_tau,
                                           random_num=self.random_num)
            logging.info(f'[event {n}, states fixed particle {particle.particle_idx}] Updating particle weight')
            particle.update_log_particle_weight(
                old_particle_weight=self.particle_weight_tensor[particle.particle_idx], n=n)

    def _generate_status_for_hyperparameter_fixed_particles(self, n: int):
        """generate status for hyperparameter fixed particles

        Args:
            n (int): _description_
        """
        for particle in self.particle_list:
            logging.info(f'[event {n}, params fixed particle {particle.particle_idx}] Sampling particle status')
            if n == 1:
                particle.sample_particle_first_event_status(particle.lambda0, particle.beta, particle.tau)
            else:
                particle.sample_particle_following_event_status(n=n)
            logging.info(f'[event {n}, params fixed particle {particle.particle_idx}] Updating particle weight')
            particle.update_log_particle_weight(
                old_particle_weight=self.particle_weight_tensor[particle.particle_idx], n=n)

    def _generate_status_for_particles(self, n: int):
        """
        :param n: event
        :return:
        """
        for particle in self.particle_list:
            logging.info(f'[event {n}, particle {particle.particle_idx}] Sampling particle status')
            if n == 1:
                particle.sample_particle_first_event_status(self.lambda0, self.beta, self.tau)
            else:
                particle.sample_particle_following_event_status(n=n)
            logging.info(f'[event {n}, particle {particle.particle_idx}] Updating hyper-parameter')
            particle.update_hyperparameter(n=n,
                                           alpha_lambda0=self.alpha_lambda0,
                                           alpha_beta=self.alpha_beta,
                                           alpha_tau=self.alpha_tau,
                                           fix_beta=self.fix_beta,
                                           fix_tau=self.fix_tau,
                                           random_num=self.random_num)
            logging.info(f'[event {n}, particle {particle.particle_idx}] Updating particle weight')
            particle.update_log_particle_weight(
                old_particle_weight=self.particle_weight_tensor[particle.particle_idx], n=n)

    def _update_particle_weight_tensor(self):
        for particle in self.particle_list:
            self.particle_weight_tensor[particle.particle_idx] = particle.log_particle_weight

    def _normalize_particle_weight_tensor(self):
        self.particle_weight_tensor = softmax(self.particle_weight_tensor, 0)

    def _reset_particle_weight_tensor(self):
        self.particle_weight_tensor = torch.tensor([1 / self.n_particle
                                                    for i in torch.arange(self.n_particle)]).to(self.device)

    def _resample_particle(self):
        resampled_particle_list = []
        ascending_particle_index = torch.argsort(self.particle_weight_tensor)
        ascending_particle_weight = self.particle_weight_tensor[ascending_particle_index]
        # construct the weight interval corresponding to the particle
        weight_interval_tensor = torch.cumsum(ascending_particle_weight, 0)
        for i in torch.arange(self.n_particle):
            u = dist.uniform.Uniform(0., 1.).sample().to(self.device)
            nearest_elem_idx = torch.argmin(torch.abs(u - weight_interval_tensor))
            if u <= weight_interval_tensor[nearest_elem_idx]:
                # remember this step need `deepcopy` because particle is a changeable object
                sampled_particle = deepcopy(self.particle_list[ascending_particle_index[nearest_elem_idx]])
            else:
                while weight_interval_tensor[nearest_elem_idx] == weight_interval_tensor[nearest_elem_idx + 1]:
                    nearest_elem_idx += 1
                sampled_particle = deepcopy(self.particle_list[ascending_particle_index[nearest_elem_idx + 1]])
            logging.info(f'resample particle {sampled_particle.particle_idx}')
            resampled_particle_list.append(sampled_particle)
        # after resampling, initialize particle weight tensor
        self._reset_particle_weight_tensor()
        # after resampling, update `self.particle_list`
        self.particle_list = resampled_particle_list
        # reset each particle's attribute: `particle_idx`
        for idx, particle in enumerate(self.particle_list):
            particle.reset_particle_index(new_index=idx)

    def filtering(self,
                  save_dir: str,
                  prefix: str = 'test',
                  rename_by_timestamp: bool = False,
                  save_res: bool = False):
        """
        filtering step in the particle filter

        Args:
            save_dir (str): dir to save filtering result
            prefix (str, optional): _description_. prefix in save path
            rename_by_timestamp (bool, optional): _description_. if rename folder by adding timestamp behind prefix
            save_res (bool, optional): _description_. if save filtering result
        """
        if rename_by_timestamp:
            time4save = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
            save_dir = f'{save_dir}/{prefix}_{time4save}'
        else:
            save_dir = f'{save_dir}/{prefix}'
        if save_res:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:  # clear old results
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
            torch.save(self.timestamp_tensor, f'{save_dir}/timestamp_tensor.pt')
            torch.save(self.text_tensor, f'{save_dir}/text_tensor.pt')
            if self.ibhp_ins:
                torch.save(self.true_lambda_tn, f'{save_dir}/true_lambda_tn.pt')
        # begin to execute filtering step
        avg_lambda0_tensor, avg_beta_tensor, avg_tau_tensor = None, None, None
        for n in torch.arange(1, self.n_sample + 1):
            if self.states_fixed:
                self._generate_status_for_states_fixed_particles(n=n)
            elif self.hyperparameter_fixed:
                self._generate_status_for_hyperparameter_fixed_particles(n=n)
            else:
                self._generate_status_for_particles(n=n)
            self._update_particle_weight_tensor()
            print(f'[event {n}] particle weight before normalizing: {self.particle_weight_tensor}')
            self._normalize_particle_weight_tensor()
            logging.info(f'[event {n}] particle weight: {self.particle_weight_tensor}')
            lambda0_particles = torch.stack([particle.lambda0 for particle in self.particle_list], 0)
            beta_particles = torch.stack([particle.beta for particle in self.particle_list], 0)
            tau_particles = torch.stack([particle.tau for particle in self.particle_list], 0)
            avg_lambda0 = self.particle_weight_tensor @ lambda0_particles
            avg_beta = self.particle_weight_tensor @ beta_particles
            avg_tau = self.particle_weight_tensor @ tau_particles
            if n == 1:
                avg_lambda0_tensor = avg_lambda0.unsqueeze(-1)
                avg_beta_tensor = avg_beta.clone()
                avg_tau_tensor = avg_tau.clone()
            else:
                avg_lambda0_tensor = torch.hstack([avg_lambda0_tensor, avg_lambda0])
                avg_beta_tensor = torch.vstack([avg_beta_tensor, avg_beta])
                avg_tau_tensor = torch.vstack([avg_tau_tensor, avg_tau])
            logging.info(f'[event {n}] lambda0: {avg_lambda0}')
            logging.info(f'[event {n}] beta: {avg_beta}')
            logging.info(f'[event {n}] tau: {avg_tau}')
            if save_res:
                torch.save(avg_lambda0_tensor, f'{save_dir}/avg_lambda0_tensor.pt')
                torch.save(avg_beta_tensor, f'{save_dir}/avg_beta_tensor.pt')
                torch.save(avg_tau_tensor, f'{save_dir}/avg_tau_tensor.pt')
                torch.save(self.particle_weight_tensor, f'{save_dir}/particle_weight_tensor.pt')
                for particle in self.particle_list:
                    if not os.path.exists(f'{save_dir}/particle-{particle.particle_idx}'):
                        os.makedirs(f'{save_dir}/particle-{particle.particle_idx}')
                    torch.save(particle.c, f'{save_dir}/particle-{particle.particle_idx}/c.pt')
                    torch.save(particle.w, f'{save_dir}/particle-{particle.particle_idx}/w.pt')
                    torch.save(particle.v, f'{save_dir}/particle-{particle.particle_idx}/v.pt')
                    torch.save(particle.lambda_tn_tensor,
                               f'{save_dir}/particle-{particle.particle_idx}/lambda_tn_tensor.pt')
                    torch.save(particle.lambda_k_tensor,
                               f'{save_dir}/particle-{particle.particle_idx}/lambda_k_tensor_event_{n}.pt')
            # resampling
            n_eff = 1 / torch.sum(torch.square(self.particle_weight_tensor))
            if n_eff <= 0.8 * self.n_particle:
                self._resample_particle()
            print('\n')


if __name__ == '__main__':
    # parameters for simulation of IBHP
    n_sample = 500
    sum_kernel_num = 3
    vocab_size = 1000
    doc_len = 20
    sim_lambda0 = 2.
    sim_beta = [1., 2., 3.]
    sim_tau = [.1, .2, .3]
    random_seed = None
    save_sim_res = True
    save_sim_dir = './model_result/simulation_data'

    # parameters for filtering
    pf_lambda0 = 3.
    pf_beta = [3., 3., 3.]
    pf_tau = [.2, .2, .2]
    pf_alpha_lambda0 = 3.
    pf_alpha_beta = 4.
    pf_alpha_tau = 1.5
    random_num_sample = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunk = False

    ibhp = IBHPTorch(n_sample=n_sample,
                     sum_kernel_num=sum_kernel_num,
                     word_num=vocab_size,
                     doc_length=doc_len,
                     lambda0=torch.tensor(sim_lambda0),
                     beta=torch.tensor(sim_beta),
                     tau=torch.tensor(sim_tau),
                     random_seed=random_seed)
    ibhp.generate_data(save_result=save_sim_res, save_path=save_sim_dir)
    ibhp.plot_simulation_c_matrix()
    word_corpus = torch.arange(vocab_size)

    # states fixed particle
    pf = ParticleFilter(n_particle=100,
                        n_sample=n_sample,
                        word_corpus=word_corpus,
                        sum_kernel_num=sum_kernel_num,
                        lambda0=torch.tensor(pf_lambda0),
                        beta=torch.tensor(pf_beta),
                        tau=torch.tensor(pf_tau),
                        alpha_lambda0=torch.tensor(pf_alpha_lambda0),
                        alpha_beta=torch.tensor(pf_alpha_beta),
                        alpha_tau=torch.tensor(pf_alpha_tau),
                        random_num=random_num_sample,
                        ibhp_ins=ibhp,
                        device=device,
                        chunk=chunk)
    pf.filtering(save_dir='./model_result/test_tensor_result', prefix='test', save_res=True, rename_by_timestamp=True)
