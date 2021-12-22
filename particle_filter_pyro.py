#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection GrazieInspection,SpellCheckingInspection
"""
@File    :   particle_filter_pyro.py
@Time    :   2021/12/15 4:10 PM
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None

This file is how to use SMCFilter algorithm with IBHP model in paper:
The Indian Buffet Hawkes Process to Model Evolving Latent Inﬂuences.
"""

import logging

import torch
import pyro
import pyro.distributions as dist_pyro

from pyro.infer import SMCFilter

# 日志输出的配置
# noinspection SpellCheckingInspection
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d--%H:%M:%S")


# noinspection SpellCheckingInspection
class IBHP_Model:
    """
    IBHP model for each particle
    """
    def __init__(self, word_corpus: torch.Tensor, base_kernel_num: int = 3):
        """
        basic params in IBHP model
        :param word_corpus:
        :param base_kernel_num:
        """
        self.base_kernel_num = base_kernel_num
        self.word_corpus = word_corpus
        self.word_num = word_corpus.shape[0]
        self.all_kappa = None
        self.lambda_0 = 1
        self.beta = torch.Tensor([2, 2, 2])
        self.tau = torch.Tensor([0.2, 0.2, 0.2])
        self.w_0 = torch.Tensor([1 / self.base_kernel_num] * self.base_kernel_num)
        self.v_0 = torch.Tensor([1 / self.word_num] * self.word_num)
        self.t = 0

    def init(self, particle_state: dict):
        """
        generate initial states for each particle
        :param particle_state:
        :return:
        """
        particle_state['K'] = pyro.sample(f'K_{self.t + 1}')