#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_likelihood.py
@Time    :   2022/05/18 15:52:27
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""

import logging
from functools import partial

import torch
from functorch import vmap
from matplotlib import cm
from matplotlib import pyplot as plt
from tqdm import tqdm

from IBHP_simulation_torch import IBHPTorch
from particle_torch import StatesFixedParticle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_log_hawkes_likelihood():
    n_sample = 500
    ibhp = IBHPTorch(n_sample=n_sample,
                     doc_length=20,
                     word_num=1000,
                     sum_kernel_num=3,
                     lambda0=torch.tensor(2.),
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
                                   device=DEVICE,
                                   ibhp=ibhp)

    lh = particle.log_hawkes_likelihood_overall(n_sample, ibhp.lambda0.to(DEVICE), ibhp.beta.to(DEVICE),
                                            ibhp.tau.to(DEVICE))

    # test likelihood
    lingspace_num = 500

    # lambda0
    lambda0_ts = torch.linspace(0., 25., lingspace_num).to(DEVICE)
    log_likelihood_vfunc = vmap(partial(particle.log_hawkes_likelihood_overall, n_sample), in_dims=(0, None, None))
    revised_lh = log_likelihood_vfunc(lambda0_ts, ibhp.beta.to(DEVICE), ibhp.tau.to(DEVICE))
    fig, ax = plt.subplots(dpi=400)
    ax.scatter(x=ibhp.lambda0, y=lh.cpu().numpy(), label='original', marker='*', color='r')
    ax.plot(lambda0_ts.cpu().numpy(), revised_lh.cpu().numpy(), label='revised', color='b')
    ax.set_xlabel(r'$\lambda_0$')
    ax.set_ylabel(r'log likelihood')
    ax.legend()
    fig.tight_layout()
    fig_name = 'test_log_hawkes_likelihood_lambda0.png'
    fig.savefig(f'./img/{fig_name}')
    logging.info(f'./img/{fig_name} saved')
    plt.close('all')

    # beta
    beta_ts = torch.linspace(0, 20, lingspace_num)
    beta_mat = ibhp.beta.repeat(lingspace_num, 1)
    tau_mat = ibhp.tau.repeat(lingspace_num, 1)
    lambda_ts = ibhp.lambda0.repeat(lingspace_num)
    fig, ax = plt.subplots(particle.sum_kernel_num.item(), 1, dpi=400)
    log_likelihood_vfunc = vmap(partial(particle.log_hawkes_likelihood_overall, n_sample), in_dims=(0, 0, 0))
    if particle.sum_kernel_num == 1:
        ax = [ax]
    for idx in range(particle.sum_kernel_num.item()):
        beta_changed = beta_mat.clone()
        beta_changed[:, idx] = beta_ts
        revised_lh = log_likelihood_vfunc(lambda_ts.to(DEVICE), beta_changed.to(DEVICE), tau_mat.to(DEVICE))
        ax[idx].scatter(x=ibhp.beta[idx], y=lh.cpu(), label='original', marker='*', color='r')
        ax[idx].plot(beta_ts.cpu(), revised_lh.cpu(), label='revised', color='b')
        ax[idx].set_xlabel(r'$\beta_{}$'.format(idx))
        ax[idx].set_ylabel(r'log likelihood')
        ax[idx].legend()
    fig.tight_layout()
    fig_name = 'test_log_hawkes_likelihood_beta.png'
    fig.savefig(f'./img/{fig_name}')
    logging.info(f'./img/{fig_name} saved')
    plt.close('all')

    # tau
    tau_ts = torch.linspace(0, 1, lingspace_num)
    tau_mat = ibhp.tau.repeat(lingspace_num, 1)
    beta_mat = ibhp.beta.repeat(lingspace_num, 1)
    lambda_ts = ibhp.lambda0.repeat(lingspace_num)
    fig, ax = plt.subplots(particle.sum_kernel_num.item(), 1, dpi=400)

    log_likelihood_vfunc = vmap(partial(particle.log_hawkes_likelihood_overall, n_sample), in_dims=(0, 0, 0))
    if particle.sum_kernel_num == 1:
        ax = [ax]
    for idx in range(particle.sum_kernel_num.item()):
        tau_changed = tau_mat.clone()
        tau_changed[:, idx] = tau_ts
        revised_lh = log_likelihood_vfunc(lambda_ts.to(DEVICE), beta_mat.to(DEVICE), tau_changed.to(DEVICE))
        ax[idx].scatter(x=ibhp.tau[idx], y=lh.cpu(), label='original', marker='*', color='r')
        ax[idx].plot(tau_ts.cpu(), revised_lh.cpu(), label='revised', color='b')
        ax[idx].set_xlabel(r'$\tau_{}$'.format(idx))
        ax[idx].set_ylabel(r'log likelihood')
        ax[idx].legend()
    fig.tight_layout()
    fig_name = 'test_log_hawkes_likelihood_tau.png'
    fig.savefig(f'./img/{fig_name}')
    logging.info(f'./img/{fig_name} saved')
    plt.close('all')

    # 3-d plot for beta and lambda0
    chunk_size = 50
    beta_ts = torch.linspace(.1, 15., lingspace_num)
    lambda0_ts = torch.linspace(.1, 10., lingspace_num)
    lambda_beta_prod = torch.cartesian_prod(lambda0_ts, beta_ts)
    beta_mat = ibhp.beta.repeat(lambda_beta_prod.shape[0], 1)
    tau_mat = ibhp.tau.repeat(lambda_beta_prod.shape[0], 1)
    fig, ax = plt.subplots(1, particle.sum_kernel_num.item(), dpi=400, subplot_kw={"projection": "3d"})
    log_likelihood_vfunc = vmap(partial(particle.log_hawkes_likelihood_overall, n_sample), in_dims=(0, 0, 0))
    if particle.sum_kernel_num == 1:
        ax = [ax]
    for idx in range(particle.sum_kernel_num.item()):
        beta_changed = beta_mat.clone()
        beta_changed[:, idx] = lambda_beta_prod[:, 1]
        lambda0_ = lambda_beta_prod[:, 0]
        if chunk_size >= 2:
            lambda0_tp = lambda0_.chunk(chunk_size)
            beta_changed_tp = beta_changed.chunk(chunk_size)
            tau_tp = tau_mat.chunk(chunk_size)
            res = []
            logging.info('Begin to calculate log likelihood')
            for i in tqdm(torch.arange(chunk_size)):
                revised_lh = log_likelihood_vfunc(lambda0_tp[i].to(DEVICE), beta_changed_tp[i].to(DEVICE),
                                                  tau_tp[i].to(DEVICE))
                res.append(revised_lh)
            revised_lh = torch.cat(res)
        else:
            revised_lh = log_likelihood_vfunc(lambda0_.to(DEVICE), beta_changed.to(DEVICE), tau_mat.to(DEVICE))
        surf = ax[idx].plot_trisurf(lambda0_,
                                    beta_changed[:, idx],
                                    revised_lh.cpu(),
                                    label='revised',
                                    cmap=cm.coolwarm)
        ax[idx].scatter(ibhp.lambda0, ibhp.beta[idx], lh.cpu(), label='original', marker='*', color='b')
        ax[idx].set_xlabel(r'$\lambda_0$')
        ax[idx].set_ylabel(r'$\beta_{}$'.format(idx))
        ax[idx].set_zlabel(r'log likelihood')
        # fig.colorbar(surf, shrink=0.6, location='bottom')
    fig.tight_layout()
    fig.savefig(f'./img/test_log_hawkes_likelihood_beta_lambda0.png')
    logging.info(f'./img/test_log_hawkes_likelihood_beta_lambda0.png saved')


if __name__ == '__main__':
    test_log_hawkes_likelihood()
