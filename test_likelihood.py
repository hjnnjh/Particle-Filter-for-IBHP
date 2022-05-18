#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_likelihood.py
@Time    :   2022/05/18 15:52:27
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""

from unittest import TestCase

import torch

from IBHP_simulation_torch import IBHPTorch
from particle_torch import Particle, StatesFixedParticle
from particle_torch import DEVICE0

# noinspection SpellCheckingInspection,DuplicatedCode


# noinspection SpellCheckingInspection
def test_log_hawkes_likelihood():
    n_sample = 500
    ibhp = IBHPTorch(n_sample=n_sample,
                     doc_len=20,
                     word_num=1000,
                     sum_kernel_num=3,
                     lambda0=2.,
                     beta=torch.tensor([1., 2., 3.]),
                     tau=torch.tensor([.3, .2, .1]),
                     random_seed=10)
    ibhp.generate_data()
    word_corpus = torch.arange(1000)
    particle = StatesFixedParticle(word_corpus=word_corpus,
                                   particle_idx=1,
                                   sum_kernel_num=3,
                                   lambda0=torch.tensor(2.),
                                   beta=torch.tensor([1., 2., 3.]),
                                   tau=torch.tensor([.3, .2, .1]),
                                   chunk=False,
                                   ibhp=ibhp)

    lh = particle.log_hawkes_likelihood(n_sample,
                                        torch.tensor(ibhp.lambda0).to(DEVICE0), ibhp.beta.to(DEVICE0),
                                        ibhp.tau.to(DEVICE0))
    large_beta_lh = particle.log_hawkes_likelihood(n_sample,
                                                   torch.tensor(2.).to(DEVICE0),
                                                   torch.tensor([1., 10., 3.]).to(DEVICE0),
                                                   torch.tensor([.3, .2, .1]).to(DEVICE0))
    print(f'real lh: {lh}')
    print(f'revised beta lh: {large_beta_lh}')


if __name__ == '__main__':
    test_log_hawkes_likelihood()