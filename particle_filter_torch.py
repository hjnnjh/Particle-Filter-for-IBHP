#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   particle_filter_torch.py
@Time    :   2022/4/14 17:52
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
import os
import time
from copy import deepcopy
from datetime import datetime
import torch
import torch.distributions as dist
from torch.nn.functional import softmax
from IBHP_simulation_torch import IBHPTorch
from particle_torch import Particle, StatesFixedParticle, TENSOR, DEVICE0


# noinspection PyShadowingNames,DuplicatedCode,PyUnresolvedReferences
class ParticleFilter:
    def __init__(self,
                 n_particle: int,
                 n_sample: int,
                 word_corpus: TENSOR,
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
                 device: torch.device = DEVICE0,
                 states_fixed: bool = False,
                 chunk: bool = False):
        """
        :param n_particle:
        :param n_sample:
        :param word_corpus:
        :param timestamp_tensor:
        :param text_tensor:
        :param true_lambda_tn:
        :param fix_w_v:
        :param simulation_w:
        :param simulation_v:
        :param lambda0:
        :param beta:
        :param tau:
        :param alpha_lambda0:
        :param alpha_beta:
        :param alpha_tau:
        :param random_num:
        :param device:
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
        if self.ibhp_ins:
            self.true_lambda_tn = self.ibhp_ins.lambda_tn_tensor
            self.timestamp_tensor = self.ibhp_ins.timestamp_tensor
            self.text_tensor = self.ibhp_ins.text
        # `self.text_tensor` and `self.timestamp_tensor` should not be `None`
        assert isinstance(self.timestamp_tensor, TENSOR)
        assert isinstance(self.text_tensor, TENSOR)
        if states_fixed:
            self.particle_list = [
                StatesFixedParticle(
                    ibhp=self.ibhp_ins,
                    particle_idx=i,
                    word_corpus=self.word_corpus,
                    lambda0=self.lambda0,
                    beta=self.beta,
                    tau=self.tau,
                    chunk=chunk) for i in range(self.n_particle)
            ]
        else:
            self.particle_list = [
                Particle(word_corpus=self.word_corpus,
                         timestamp_tensor=self.timestamp_tensor,
                         text_tensor=self.text_tensor,
                         particle_idx=i,
                         device=self.device,
                         chunk=chunk) for i in range(self.n_particle)
            ]
        self.particle_weight_tensor = torch.tensor([
            1 / self.n_particle for i in torch.arange(self.n_particle)
        ]).to(self.device)

    def _generate_status_for_particles(self, n: int):
        """
        :param n: event
        :return:
        """
        for particle in self.particle_list:
            if hasattr(particle, 'states_fixed'):
                particle.compute_lambda_tn(n)
            else:
                logging.info(
                    f'[event {n}, particle {particle.particle_idx}] Sampling particle status'
                )
                if n == 1:
                    particle.sample_particle_first_event_status(
                        self.lambda0, self.beta, self.tau)
                else:
                    particle.sample_particle_following_event_status(n=n)
            if hasattr(particle, 'states_fixed'):
                logging.info(
                    f'[event {n}, states fixed particle {particle.particle_idx}] Updating hyper-parameter'
                )
            else:
                logging.info(
                    f'[event {n}, particle {particle.particle_idx}] Updating hyper-parameter'
                )
            particle.update_hyperparameter(n=n,
                                           alpha_lambda0=self.alpha_lambda0,
                                           alpha_beta=self.alpha_beta,
                                           alpha_tau=self.alpha_tau,
                                           random_num=self.random_num)

            if hasattr(particle, 'states_fixed'):
                logging.info(
                    f'[event {n}, states fixed particle {particle.particle_idx}] Updating particle weight'
                )
            else:
                logging.info(
                    f'[event {n}, particle {particle.particle_idx}] Updating particle weight'
                )
            particle.update_log_particle_weight(
                old_particle_weight=self.particle_weight_tensor[
                    particle.particle_idx],
                n=n)

    def _update_particle_weight_tensor(self):
        for particle in self.particle_list:
            self.particle_weight_tensor[
                particle.particle_idx] = particle.log_particle_weight

    def _normalize_particle_weight_tensor(self):
        self.particle_weight_tensor = softmax(self.particle_weight_tensor, 0)

    def _reset_particle_weight_tensor(self):
        self.particle_weight_tensor = torch.tensor([
            1 / self.n_particle for i in torch.arange(self.n_particle)
        ]).to(self.device)

    def _resample_particle(self):
        resampled_particle_list = []
        ascending_particle_index = torch.argsort(self.particle_weight_tensor)
        ascending_particle_weight = self.particle_weight_tensor[
            ascending_particle_index]
        # construct the weight interval corresponding to the particle
        weight_interval_tensor = torch.cumsum(ascending_particle_weight, 0)
        for i in torch.arange(self.n_particle):
            u = dist.uniform.Uniform(0., 1.).sample().to(self.device)
            nearest_elem_idx = torch.argmin(
                torch.abs(u - weight_interval_tensor))
            if u <= weight_interval_tensor[nearest_elem_idx]:
                # remember this step need `deepcopy` because particle is a changeable object
                sampled_particle = deepcopy(self.particle_list[
                                                ascending_particle_index[nearest_elem_idx]])
            else:
                while weight_interval_tensor[nearest_elem_idx] == weight_interval_tensor[nearest_elem_idx + 1]:
                    nearest_elem_idx += 1
                sampled_particle = deepcopy(self.particle_list[
                                                ascending_particle_index[nearest_elem_idx + 1]])
            logging.info(f'resample particle {sampled_particle.particle_idx}')
            resampled_particle_list.append(sampled_particle)
        # after resampling, initialize particle weight tensor
        self._reset_particle_weight_tensor()
        # after resampling, update `self.particle_list`
        self.particle_list = resampled_particle_list
        # reset each particle's attribute: `particle_idx`
        for idx, particle in enumerate(self.particle_list):
            particle.reset_particle_index(new_index=idx)

    def filtering(self, username: str, save_res: bool = False):
        time4save = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
        save_dir = f'./model_result/real_data_tensor_res/model_result_{username}_{time4save}'
        if save_res:
            if not os.path.exists('./model_result/real_data_tensor_res'):
                os.mkdir('./model_result/real_data_tensor_res')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(self.timestamp_tensor,
                       f'{save_dir}/timestamp_tensor.pt')
            torch.save(self.text_tensor, f'{save_dir}/text_tensor.pt')
            if self.ibhp_ins:
                torch.save(self.true_lambda_tn, f'{save_dir}/true_lambda_tn.pt')
        # begin to execute filtering step
        avg_lambda0_tensor, avg_beta_tensor, avg_tau_tensor = None, None, None
        for n in torch.arange(1, self.n_sample + 1):
            self._generate_status_for_particles(n=n)
            self._update_particle_weight_tensor()
            self._normalize_particle_weight_tensor()
            logging.info(
                f'[event {n}] particle weight: {self.particle_weight_tensor}')
            lambda0_particles = torch.stack(
                [particle.lambda0 for particle in self.particle_list], 0)
            beta_particles = torch.stack(
                [particle.beta for particle in self.particle_list], 0)
            tau_particles = torch.stack(
                [particle.tau for particle in self.particle_list], 0)
            avg_lambda0 = self.particle_weight_tensor @ lambda0_particles
            avg_beta = self.particle_weight_tensor @ beta_particles
            avg_tau = self.particle_weight_tensor @ tau_particles
            if n == 1:
                avg_lambda0_tensor = avg_lambda0.unsqueeze(-1)
                avg_beta_tensor = avg_beta.clone()
                avg_tau_tensor = avg_tau.clone()
            else:
                avg_lambda0_tensor = torch.hstack(
                    [avg_lambda0_tensor, avg_lambda0])
                avg_beta_tensor = torch.vstack([avg_beta_tensor, avg_beta])
                avg_tau_tensor = torch.vstack([avg_tau_tensor, avg_tau])
            logging.info(f'[event {n}] lambda0: {avg_lambda0}')
            logging.info(f'[event {n}] beta: {avg_beta}')
            logging.info(f'[event {n}] tau: {avg_tau}')
            if save_res:
                torch.save(avg_lambda0_tensor,
                           f'{save_dir}/avg_lambda0_tensor.pt')
                torch.save(avg_beta_tensor, f'{save_dir}/avg_beta_tensor.pt')
                torch.save(avg_tau_tensor, f'{save_dir}/avg_tau_tensor.pt')
                torch.save(self.particle_weight_tensor,
                           f'{save_dir}/particle_weight_tensor.pt')
                for particle in self.particle_list:
                    if not os.path.exists(
                            f'{save_dir}/particle-{particle.particle_idx}'):
                        os.mkdir(
                            f'{save_dir}/particle-{particle.particle_idx}')
                    torch.save(
                        particle.c,
                        f'{save_dir}/particle-{particle.particle_idx}/c.pt')
                    torch.save(
                        particle.w,
                        f'{save_dir}/particle-{particle.particle_idx}/w.pt')
                    torch.save(
                        particle.v,
                        f'{save_dir}/particle-{particle.particle_idx}/v.pt')
                    torch.save(
                        particle.lambda_tn_tensor,
                        f'{save_dir}/particle-{particle.particle_idx}/lambda_tn_tensor.pt'
                    )
                    torch.save(
                        particle.lambda_k_tensor,
                        f'{save_dir}/particle-{particle.particle_idx}/lambda_k_tensor_event_{n}.pt'
                    )
            # resampling
            n_eff = 1 / torch.sum(torch.square(self.particle_weight_tensor))
            if n_eff < 0.8 * self.n_particle:
                self._resample_particle()


if __name__ == '__main__':
    n_sample = 1000
    ibhp = IBHPTorch(n_sample=n_sample,
                     sum_kernel_num=3,
                     word_num=1000,
                     doc_len=20,
                     lambda0=2.,
                     beta=torch.tensor([1., 2., 3.]),
                     tau=torch.tensor([.3, .2, .1]),
                     random_seed=10)
    ibhp.generate_data()
    word_corpus = torch.arange(1000)

    # totally random particle
    # pf = ParticleFilter(
    #     n_particle=10,
    #     n_sample=n_sample,
    #     word_corpus=word_corpus,
    #     timestamp_tensor=ibhp.timestamp_tensor,
    #     text_tensor=ibhp.text,
    #     true_lambda_tn=ibhp.lambda_tn_tensor,
    #     lambda0=torch.tensor(5.),
    #     beta=torch.tensor([5., 2., 3.]),
    #     tau=torch.tensor([.3, .2, .1]),
    #     fix_w_v=True,
    #     simulation_w=ibhp.w,
    #     simulation_v=ibhp.v,  # float overflow
    #     alpha_lambda0=torch.tensor(3.),
    #     alpha_beta=torch.tensor(3.),
    #     alpha_tau=torch.tensor(1.5),
    #     random_num=10000
    # )

    # states fixed particle
    pf_states_fixed_particles = ParticleFilter(n_particle=10,
                                               n_sample=n_sample,
                                               word_corpus=word_corpus,
                                               lambda0=torch.tensor(5.),
                                               beta=torch.tensor([5., 5., 5.]),
                                               tau=torch.tensor([.5, .5, .5]),
                                               alpha_lambda0=torch.tensor(3.),
                                               alpha_beta=torch.tensor(3.),
                                               alpha_tau=torch.tensor(1.5),
                                               random_num=2500,
                                               states_fixed=True,
                                               ibhp_ins=ibhp,
                                               chunk=False)
    pf_states_fixed_particles.filtering(username='test', save_res=True)
