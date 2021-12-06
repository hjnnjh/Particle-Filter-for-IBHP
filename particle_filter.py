#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
@File    :   particle_filter.py
@Time    :   2021/11/10 22:01
@Author  :   Jinnan Huang
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
from collections import Counter
from functools import partial
from multiprocessing import Pool, Manager

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as scipy_gamma_dist
from tqdm import tqdm

from IBHP_sim import IBHP


# noinspection PyPep8Naming
def transfer_multi_dist_res2vec(T: np.ndarray):
    """
    将由多项分布生成的数据矩阵转化为具体的词矩阵
    :param T:
    :return:
    """

    def transfer_occurrence_data(idx: list, data: list):
        res = []
        res.extend([[idx[i]] * data[i] for i in range(len(idx))])
        res = [i for k in res for i in k]
        return res

    word_index = np.argwhere(T != 0)
    index_list = [word_index[word_index[:, 0] == n][:, 1].tolist() for n in np.unique(word_index[:, 0])]
    word_occurrence_list = [T[i, idx].tolist() for i, idx in enumerate(index_list)]
    word_corpus_mat = np.array([transfer_occurrence_data(index_list[i], word_occurrence_list[i])
                                for i in range(len(index_list))])
    return word_corpus_mat


# noinspection SpellCheckingInspection,PyPep8Naming
def plot_mh_res(arr, title: str):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=400)
    ax.plot(np.arange(len(arr)), arr)
    ax.set_title(label=title)
    plt.show()


# noinspection PyAttributeOutsideInit,PyMissingConstructor,SpellCheckingInspection,DuplicatedCode,PyPep8Naming
class Particle(IBHP):
    """
    这个类中实现了单个粒子的采样、超参数更新、粒子权重计算的所有的步骤
    """

    def __init__(self, word_corpus: np.ndarray, L: int = 3):
        """
        :param L: base kernel的数量
        :param word_corpus: 词典
        """
        self.L = L
        self.word_corpus = word_corpus
        self.S = len(self.word_corpus)
        self.kappa_all = None

        # 测试粒子并行更新
        # self.kappa_all = np.arange(1)
        # self.kappa = np.zeros((1, 1))
        # self.new_particle_weight = 1
        # self.lambda_0 = np.zeros(1)
        # self.beta = np.zeros((1, self.L))
        # self.tau = np.zeros((1, self.L))
        # self.K = 1

    def sample_first_event_particle(self, shape_lammbda0, scale_lambda0, shape_beta, scale_beta, shape_tau, scale_tau):
        """
        生成first event对应的粒子
        :return:
        """
        # 初始化lambda_0, beta_l, tau_l
        self.lambda_0 = np.random.gamma(shape_lammbda0, scale_lambda0, 1)  # array, shape=(1, )
        self.beta = np.random.gamma(shape_beta, scale_beta, self.L).reshape(-1, self.L)  # matrix, shape=(1, L)
        self.tau = np.random.gamma(shape_tau, scale_tau, self.L).reshape(-1, self.L)  # array, shape=(1, L),

        self.w_0 = np.array([1 / self.L] * self.L)  # array, shape=(L, )
        self.v_0 = np.array([1 / self.S] * self.S)  # array, shape(S, )

        # 生成K
        self.K = np.random.poisson(self.lambda_0[-1], 1)[0]  # K: int
        while self.K == 0:
            print(f'[event 1] 初始化生成的主题数K为0，重新生成')
            self.K = np.random.poisson(self.lambda_0[-1], 1)[0]

        # 生成主题出现情况矩阵c, 令所有c_k=1
        self.c = np.ones((1, self.K))  # matrix, shape=(1, self.K)
        # 生成w
        self.w = np.random.dirichlet(self.w_0, self.K).T  # matrix, shape=(L, K), 每一列是每个k的权重
        # 生成v
        self.v = np.random.dirichlet(self.v_0, self.K).T  # matrix, shape=(S, K), 每一列是每个k的词分布

        # 计算kappa
        base_kernel_func = np.vectorize(partial(self.base_kernel_l, 0))
        self.gamma = base_kernel_func(self.beta[-1, :], self.tau[-1, :]).reshape(-1, self.L)  # shape=(n, L)
        self.kappa = np.zeros((1, self.K))
        for k in np.arange(self.K):
            self.kappa[0, k] = self.w[:, k] @ self.gamma[-1, :] * self.c[0, k]

        # update kappa_all
        self.kappa_all = self.kappa.copy().reshape(-1)

    def sample_following_event_particle(self, n: int):
        """
        生成following event对应的粒子
        :param n: 第n个event
        :return:
        """
        if n == 2:
            # 计算lambda_t1，作为self.lambda_tn_arr的第一个元素
            self.lambda_n_minus_1_arr = np.einsum('ij->j', self.kappa / np.count_nonzero(self.kappa, axis=1).
                                                  reshape(self.kappa.shape[0], -1))
            lambda_t1 = np.sum(self.lambda_n_minus_1_arr)
            self.lambda_tn_arr = np.array([lambda_t1])

        # 计算从已有K个主题中生成c的概率
        p = self.lambda_n_minus_1_arr / (self.lambda_0[-1] / self.K + self.lambda_n_minus_1_arr)
        # 生成已有K个主题的主题出现情况向量
        generate_old_c_func = np.vectorize(self.generate_c)
        c_old = generate_old_c_func(p)

        # 生成K+
        k_plus = np.random.poisson(self.lambda_0[-1] / (self.lambda_0[-1] + np.sum(self.lambda_n_minus_1_arr)),
                                   1)[0]
        # 当K+等于0时，检查c_old是不是全为0，如果全为0，则重新生成c_old
        if k_plus == 0:
            while np.all(c_old == 0):
                print(f'[event {n}]当K+为0时，c_old也全为0，重新生成c_old')
                c_old = generate_old_c_func(p)
        # 更新K
        self.K = self.K + k_plus

        if k_plus:
            # 如果K+大于0，初始化新的主题出现情况向量
            c_new = np.ones(k_plus)
            c = np.hstack((c_old, c_new))
            self.c = np.hstack((self.c, np.zeros((self.c.shape[0], k_plus))))  # 将已有的c矩阵用0补全
            self.c = np.vstack((self.c, c))  # 将新的c向量加入c矩阵

            # 如果K+大于0, 生成新的w_k, 更新self.w
            w_new = np.random.dirichlet(self.w_0, k_plus).T
            self.w = np.hstack((self.w, w_new))

            # 如果K+大于0, 生成新的v_k, 更新self.v
            v_new = np.random.dirichlet(self.v_0, k_plus).T
            self.v = np.hstack((self.v, v_new))

            # 如果K+大于0, 计算新的kappa
            kappa_old = self.kappa_all * c_old  # 根据已有的主题出现情况更新old kappa
            kappa_new = np.zeros(k_plus)
            for k in np.arange(k_plus):
                kappa_new[k] = w_new[:, k].T @ self.gamma[-1, :] * c_new[k]

            # update kappa_all
            self.kappa_all = np.hstack((self.kappa_all, kappa_new))

            # update kappa matrix
            kappa = np.hstack((kappa_old, kappa_new))
            self.kappa = np.hstack((self.kappa, np.zeros((self.kappa.shape[0], k_plus))))  # 将已有的kappa矩阵用0补全
            self.kappa = np.vstack((self.kappa, kappa))  # 将新的kappa加入kappa矩阵
        else:
            self.c = np.vstack((self.c, c_old))
            kappa_old = self.kappa_all * c_old  # 根据已有的主题出现情况更新old kappa
            self.kappa = np.vstack((self.kappa, kappa_old))

        # print(f'n={n}时主题出现情况：{self.c}\n')
        # print(f'n={n}时base kernel的权重：{self.w}\n')
        # print(f'n={n}时的factor kernel：{self.kappa}\n')
        # print(f'n={n}时主题的词分布：{self.v}\n')

        # 计算lambda_tn, 并将lambda_tn加入self.lambda_arr
        non_zero_kappa_idx = np.argwhere(self.kappa[-1, :] != 0)[:, 0]
        lambda_tn_arr = np.einsum('ij->j', self.kappa / np.count_nonzero(self.kappa, axis=1).
                                  reshape(self.kappa.shape[0], -1))
        self.lambda_n_minus_1_arr = lambda_tn_arr
        lambda_tn = np.sum(lambda_tn_arr[non_zero_kappa_idx])
        self.lambda_tn_arr = np.append(self.lambda_tn_arr, lambda_tn)

    # 计算MH算法中的似然函数，这里用的是Hawkes process的似然函数
    @staticmethod
    def log_hawkes_likelihood(lambda_0, beta_arr: np.ndarray, tau_arr: np.ndarray, w: np.ndarray,
                              kappa: np.ndarray, t_vec: np.ndarray, n: int):
        """
        hawkes对数似然函数
        :param w: base kernel的权重矩阵
        :param n: 当前的样本顺序，需大于等于1
        :param t_vec: 时间戳向量(观测数据)
        :param kappa: matrix
        :param lambda_0:
        :param beta_arr: array
        :param tau_arr: array
        :return:
        """
        lambda_all = np.zeros(n)
        for j in np.arange(n):  # j range
            kappa_neq_zero_index_j = np.argwhere(kappa[j, :] != 0)[:, 0]
            i = np.arange(j + 1)
            tj_minus_ti = t_vec[j] - t_vec[i]  # t[i]是一个array，因此tj_minus_ti也是一个array

            t_div_tau_func = np.vectorize(np.divide, signature='(),(n)->(n)')
            beta_times_tau_func = np.vectorize(lambda x, y: x * y, signature='(n),(n)->(n)')

            gamma_arr = beta_times_tau_func(beta_arr, np.exp(- t_div_tau_func(tj_minus_ti, tau_arr)))
            numerator = (w[:, kappa_neq_zero_index_j].T @ gamma_arr.T).T
            denominator = np.count_nonzero(kappa[: j + 1, :], axis=1).reshape(-1, 1)
            lambda_j = np.einsum('ij->', numerator / denominator)  # 计算lambda_j
            lambda_all[j] = lambda_j  # 保存lambda_j
        # 计算似然函数
        log_lh = -(lambda_0 + np.sum(lambda_all)) * np.sum(np.log(lambda_all))
        return log_lh

    @staticmethod
    def gamma_dist(shape, scale, x):
        """
        gamma分布的表达式，后续用作计算提议分布和先验分布的p值
        :param shape:
        :param scale:
        :param x: 数据
        :return:
        """
        # p = (x ** (shape - 1)) * (np.exp(-(x / scale)) / (scale ** shape * gamma(shape)))
        return scipy_gamma_dist.pdf(x=x, a=shape, scale=scale)

    def update_lambda_0(self, old_lambda_0, proposal_shape, proposal_scale, prior_shape, prior_scale, beta, tau, t,
                        sample_order):
        """
        使用MH算法
        :param prior_scale:
        :param prior_shape:
        :param old_lambda_0:
        :param proposal_shape:
        :param proposal_scale:
        :param beta:
        :param tau:
        :param t:
        :param sample_order:
        :return:
        """
        # 从提议分布中抽取一个新的lambda_0
        new_lambda_0 = np.random.gamma(shape=proposal_shape, scale=proposal_scale, size=1)[0]

        old_log_like_lambda0 = self.log_hawkes_likelihood(lambda_0=old_lambda_0, beta_arr=beta, tau_arr=tau, w=self.w,
                                                          kappa=self.kappa, t_vec=t, n=sample_order)
        new_log_like_lambda0 = self.log_hawkes_likelihood(lambda_0=new_lambda_0, beta_arr=beta, tau_arr=tau, w=self.w,
                                                          kappa=self.kappa, t_vec=t, n=sample_order)

        old_prior_lambda0 = self.gamma_dist(shape=prior_shape, scale=prior_scale, x=old_lambda_0)
        new_prior_lambda0 = self.gamma_dist(shape=prior_shape, scale=prior_scale, x=new_lambda_0)

        old_proposal_lambda0 = self.gamma_dist(shape=proposal_shape, scale=proposal_scale,
                                               x=new_lambda_0)  # q(old, new), old作为参数
        new_proposal_lambda0 = self.gamma_dist(shape=proposal_shape, scale=proposal_scale,
                                               x=old_lambda_0)  # q(new, old), new作为参数

        # 计算接受率的对数
        log_accept_ratio_lambda0 = np.log(new_prior_lambda0) - np.log(old_prior_lambda0) \
                                   + new_log_like_lambda0 - old_log_like_lambda0 \
                                   + np.log(new_proposal_lambda0) - np.log(old_proposal_lambda0)

        # 判断是否接受样本new_lambda_0
        u_lambda_0 = np.random.rand()
        if u_lambda_0 == 0 or np.log(u_lambda_0) <= log_accept_ratio_lambda0:
            return new_lambda_0
        else:
            return old_lambda_0

    def update_beta_l(self, old_beta_l, proposal_shape, proposal_scale, prior_shape, prior_scale,
                      lambda_0, beta, tau,
                      t, beta_l_index, sample_order):
        """
        更新Beta_l
        :param old_beta_l:
        :param proposal_shape:
        :param proposal_scale:
        :param prior_shape:
        :param prior_scale:
        :param lambda_0:
        :param beta:
        :param tau:
        :param t:
        :param beta_l_index: beta_l的index
        :param sample_order:
        :return:
        """
        # 提议一个新的beta_l
        new_beta_l = np.random.gamma(shape=proposal_shape, scale=proposal_scale, size=1)[0]

        old_log_like_beta_l = self.log_hawkes_likelihood(lambda_0=lambda_0, beta_arr=beta, tau_arr=tau,
                                                         w=self.w, kappa=self.kappa, t_vec=t, n=sample_order)

        # 生成new_beta_l后更新对应位置的beta_la[l] = new_beta_l
        beta[beta_l_index] = new_beta_l
        new_log_like_beta_l = self.log_hawkes_likelihood(lambda_0=lambda_0, beta_arr=beta, tau_arr=tau,
                                                         w=self.w, kappa=self.kappa, t_vec=t, n=sample_order)

        old_prior_beta_l = self.gamma_dist(shape=prior_shape, scale=prior_scale, x=old_beta_l)
        new_prior_beta_l = self.gamma_dist(shape=prior_shape, scale=prior_scale, x=new_beta_l)

        old_proposal_beta_l = self.gamma_dist(shape=proposal_shape, scale=proposal_scale,
                                              x=new_beta_l)  # q(old, new), old作为参数
        new_proposal_beta_l = self.gamma_dist(shape=proposal_shape, scale=proposal_scale,
                                              x=old_beta_l)  # q(new, old), new作为参数

        log_accept_ratio_beta_l = np.log(new_prior_beta_l) - np.log(old_prior_beta_l) + new_log_like_beta_l - \
                                  old_log_like_beta_l + np.log(new_proposal_beta_l) - np.log(old_proposal_beta_l)

        u_beta_l = np.random.rand()
        if u_beta_l == 0 or np.log(u_beta_l) <= log_accept_ratio_beta_l:
            return new_beta_l
        else:
            return old_beta_l

    def update_tau_l(self, old_tau_l, proposal_shape, proposal_scale, prior_shape, prior_scale,
                     lambda_0, beta, tau,
                     t, tau_l_index, sample_order):
        # 提议一个新的tau_l
        new_tau_l = np.random.gamma(shape=proposal_shape, scale=proposal_scale, size=1)[0]
        # 生成new_tau_l后更新对应位置的beta_l
        old_log_like_tau_l = self.log_hawkes_likelihood(lambda_0=lambda_0, beta_arr=beta,
                                                        tau_arr=tau,
                                                        w=self.w, kappa=self.kappa, t_vec=t, n=sample_order)
        # 生成new_tau_l后更新对应位置的tau_l
        tau[tau_l_index] = new_tau_l
        new_log_like_tau_l = self.log_hawkes_likelihood(lambda_0=lambda_0, beta_arr=beta,
                                                        tau_arr=tau, w=self.w,
                                                        kappa=self.kappa, t_vec=t, n=sample_order)

        old_prior_tau_l = self.gamma_dist(shape=prior_shape, scale=prior_scale, x=old_tau_l)
        new_prior_tau_l = self.gamma_dist(shape=prior_shape, scale=prior_scale, x=new_tau_l)

        old_proposal_tau_l = self.gamma_dist(shape=proposal_shape, scale=proposal_scale,
                                             x=new_tau_l)  # q(old, new), old作为参数
        new_proposal_tau_l = self.gamma_dist(shape=proposal_shape, scale=proposal_scale,
                                             x=old_tau_l)  # q(new, old), new作为参数

        log_accept_ratio_tau_l = np.log(new_prior_tau_l) - np.log(old_prior_tau_l) + \
                                 new_log_like_tau_l - old_log_like_tau_l + \
                                 np.log(new_proposal_tau_l) - np.log(old_proposal_tau_l)

        u_tau_l = np.random.rand()
        if u_tau_l == 0 or np.log(u_tau_l) <= log_accept_ratio_tau_l:
            return new_tau_l
        else:
            return old_tau_l

    # noinspection PyUnboundLocalVariable
    def sample_hyperparams(self,
                           lambda0_proposal_shape, lambda0_proposal_scale, lambda0_prior_shape, lambda0_prior_scale,
                           beta_proposal_shape, beta_proposal_scale, beta_prior_shape, beta_prior_scale,
                           tau_proposal_shape, tau_proposal_scale, tau_prior_shape, tau_prior_scale,
                           t, event_order: int, particle_idx: int, n_iter: int = 5000, burning: int = 4000):
        """
        使用MH算法更新每个粒子的lambda_0, beta, tau
        :param tau_prior_scale: tau先验分布的scale参数
        :param tau_prior_shape: tau先验分布的shape参数
        :param tau_proposal_scale: tau提议分布的scale参数
        :param tau_proposal_shape: tau提议分布的shape参数
        :param beta_prior_scale: beta先验分布的scale参数
        :param beta_prior_shape: beta先验分布的shape参数
        :param beta_proposal_scale: beta提议分布的scale参数
        :param beta_proposal_shape: beta提议分布的shape参数
        :param lambda0_prior_scale: lambda0先验分布的scale参数
        :param lambda0_prior_shape: lambda0先验分布的shape参数
        :param lambda0_proposal_scale: lambda0提议分布的scale参数
        :param lambda0_proposal_shape: lambda0提议分布的shape参数
        :param t: 时间戳向量（观测数据）
        :param event_order: 当前是第几个样本（样本顺序）
        :param n_iter: MH算法迭代次数
        :param burning: 燃烧期的样本数
        :param particle_idx: 粒子的序号
        :return:
        """
        lambda_0_candi_list = []
        beta_candi_list = [[] for i in range(self.L)]
        tau_candi_list = [[] for i in range(self.L)]
        beta = self.beta.copy().reshape(-1)
        tau = self.tau.copy().reshape(-1)
        n_iter_range = tqdm(np.arange(n_iter))
        for i in n_iter_range:
            # 采样lambda_0
            n_iter_range.set_description(f"[event {event_order}, particle {particle_idx + 1}] Sampling hyperparameters")
            if i == 0:
                new_lambda_0 = self.update_lambda_0(old_lambda_0=self.lambda_0[-1],
                                                    proposal_shape=lambda0_proposal_shape,
                                                    proposal_scale=lambda0_proposal_scale,
                                                    prior_scale=lambda0_prior_scale, prior_shape=lambda0_prior_shape,
                                                    beta=beta, tau=tau, t=t, sample_order=event_order)
            else:
                new_lambda_0 = self.update_lambda_0(old_lambda_0=old_lambda_0,
                                                    proposal_shape=lambda0_proposal_shape,
                                                    proposal_scale=lambda0_proposal_scale,
                                                    prior_scale=lambda0_prior_scale, prior_shape=lambda0_prior_shape,
                                                    beta=beta, tau=tau, t=t, sample_order=event_order)
            lambda_0_candi_list.append(new_lambda_0)
            old_lambda_0 = new_lambda_0

            # 采样beta_l
            for beta_l_index in np.arange(self.L):
                new_beta_l = self.update_beta_l(old_beta_l=beta[beta_l_index],
                                                proposal_shape=beta_proposal_shape, proposal_scale=beta_proposal_scale,
                                                prior_shape=beta_prior_shape, prior_scale=beta_prior_scale,
                                                lambda_0=new_lambda_0, beta=beta, tau=tau, t=t,
                                                beta_l_index=beta_l_index, sample_order=event_order)
                beta[beta_l_index] = new_beta_l
                beta_candi_list[beta_l_index].append(new_beta_l)

            # 采样tau_l
            for tau_l_index in np.arange(self.L):
                new_tau_l = self.update_tau_l(old_tau_l=tau[tau_l_index],
                                              proposal_shape=tau_proposal_shape,
                                              proposal_scale=tau_proposal_scale,
                                              prior_shape=tau_prior_shape, prior_scale=tau_prior_scale,
                                              lambda_0=new_lambda_0, beta=beta, tau=tau, t=t,
                                              tau_l_index=tau_l_index, sample_order=event_order)
                tau[tau_l_index] = new_tau_l
                tau_candi_list[tau_l_index].append(new_tau_l)

        # 丢弃燃烧期的样本
        lambda_0_candi_list = lambda_0_candi_list[burning:]
        for l in np.arange(self.L):
            beta_candi_list[l] = beta_candi_list[l][burning:]
            tau_candi_list[l] = tau_candi_list[l][burning:]
        return lambda_0_candi_list, beta_candi_list, tau_candi_list

    def update_trigger_kernel(self, lambda_0_candi_list, beta_candi_list, tau_candi_list):
        """
        更新trigger_kernel
        :param lambda_0_candi_list: 丢弃燃烧期样本之后的候选lambda0列表
        :param beta_candi_list: 丢弃燃烧期样本之后的候选beta列表
        :param tau_candi_list: 丢弃燃烧期样本之后的候选tau列表
        :return:
        """
        self.lambda_0 = np.append(self.lambda_0, np.mean(lambda_0_candi_list))
        self.beta = np.array(beta_candi_list).mean(axis=1).reshape(1, -1)
        self.tau = np.array(tau_candi_list).mean(axis=1).reshape(1, -1)

    def update_paricle_weight(self, old_particle_weight, t: np.ndarray, T: np.ndarray, event_order: int):
        """
        更新粒子权重

        :param old_particle_weight:
        :param T: 文本向量
        :param t: 时间戳向量
        :param event_order: 当前是第几个样本（样本顺序），应该是大于等于1的值
        :return:
        """
        # 计算lambda_tn的似然函数
        kappa_neq_zero_index_n = np.argwhere(self.kappa[event_order - 1, :] != 0)[:, 0]
        i_range = np.arange(event_order)
        tn_minus_ti = t[event_order - 1] - t[i_range]  # t[i_range]是一个array，因此tj_minus_ti也是一个array

        t_div_tau_func = np.vectorize(np.divide, signature='(),(n)->(n)')
        beta_times_tau_func = np.vectorize(lambda x, y: x * y, signature='(n),(n)->(n)')

        gamma_arr = beta_times_tau_func(self.beta, np.exp(- t_div_tau_func(tn_minus_ti, self.tau)))
        numerator = (self.w[:, kappa_neq_zero_index_n].T @ gamma_arr.T).T
        denominator = np.count_nonzero(self.kappa[:event_order, :], axis=1).reshape(-1, 1)
        lambda_tn = np.einsum('ij->', numerator / denominator)  # 计算lambda_tn
        if event_order == 1:
            likelihood_timestamp = lambda_tn * np.exp(-lambda_tn * t[event_order - 1])
        else:
            likelihood_timestamp = lambda_tn * np.exp(-lambda_tn * (t[event_order - 1] - t[event_order - 2]))

        # 计算文本的似然函数
        vn_avg = np.einsum('ij->i', self.v[:, kappa_neq_zero_index_n]) / np.count_nonzero(
            self.kappa[event_order - 1, :])
        Tn = transfer_multi_dist_res2vec(T)[event_order - 1, :]  # shape=(S, )
        count_dict = Counter(Tn)
        likelihood_text = 1
        for k, v in count_dict.items():
            likelihood_text = likelihood_text * (vn_avg[k] ** v)
        print(f'[event {event_order}] old_particle_weight: {old_particle_weight}')
        # 计算更新后的粒子权重
        self.new_particle_weight = old_particle_weight * likelihood_timestamp * likelihood_text


# noinspection PyPep8Naming,SpellCheckingInspection,PyAttributeOutsideInit
class Particle_Filter:
    """
    这个类控制所有粒子的权重更新、归一化，实现粒子的重采样（采用并行的方式）
    """

    def __init__(self, t: np.ndarray, T: np.ndarray, n_particle: int, word_corpus: np.ndarray, L: int = 3):
        """
        生成粒子以及初始化粒子权重

        :param t: 时间戳向量
        :param T: 文本
        :param n_particle: 粒子个数
        :param word_corpus: 词典
        :param L: base kernel的个数
        """
        assert len(t) == T.shape[0]
        self.t = t
        self.T = T
        self.n_sample = self.T.shape[0]
        self.n_particle = n_particle
        self.word_corpus = word_corpus
        self.particle_list = [Particle(word_corpus=word_corpus, L=L) for i in np.arange(self.n_particle)]
        self.particle_weight_list = np.array([1 / self.n_particle] * self.n_particle)

    def normalize_particle_weight(self):
        """
        归一化粒子权重（将权重映射到0-1区间，并且和为1）
        :return:
        """
        self.particle_weight_list = self.particle_weight_list / np.sum(self.particle_weight_list)

    def resample_particles(self):
        """
        粒子重采样, 重新采样n_particle个粒子
        :return:
        """
        new_particle_list = []
        sorted_particle_index = np.argsort(self.particle_weight_list)  # 得到的是粒子权重升序排列的索引
        sorted_particle_weight = self.particle_weight_list[sorted_particle_index]  # 升序排列后的粒子权重
        # 构造粒子对应的权重区间
        for i in np.arange(self.n_particle - 1, -1, -1):
            sorted_particle_weight[i] = np.sum(sorted_particle_weight[: i + 1])
        # 重新采样n_particle个粒子
        for i in np.arange(self.n_particle):
            u = np.random.rand()
            nearest_elem_idx = np.argmin(np.abs(u - sorted_particle_weight))
            if u <= sorted_particle_weight[nearest_elem_idx]:
                new_particle = self.particle_list[nearest_elem_idx]
            else:
                new_particle = self.particle_list[nearest_elem_idx + 1]
            new_particle_list.append(new_particle)
        # 更新粒子列表，重新设置粒子权重
        self.particle_list = new_particle_list
        self.particle_weight_list = np.array([1 / self.n_particle] * self.n_particle)

    def generate_first_event4paricle(self, particle_idx_pair: (int, Particle)):
        """
        给每个粒子生成第一个event对应的状态
        :param particle_idx_pair:
        :return:
        """
        particle_idx = particle_idx_pair[0]
        particle = particle_idx_pair[1]
        print(f'[event 1, paricle {particle_idx + 1}] Begin to sample particle……')
        particle.sample_first_event_particle(shape_lammbda0=2, scale_lambda0=2, shape_beta=2, scale_beta=2,
                                             shape_tau=2, scale_tau=2)
        lambda_0_candi_list, beta_candi_list, tau_candi_list = particle.sample_hyperparams(
            lambda0_proposal_shape=3,
            lambda0_proposal_scale=2,
            lambda0_prior_shape=2,
            lambda0_prior_scale=2,
            beta_proposal_shape=3,
            beta_proposal_scale=2,
            beta_prior_shape=2,
            beta_prior_scale=2,
            tau_proposal_shape=3,
            tau_proposal_scale=2,
            tau_prior_shape=2,
            tau_prior_scale=2,
            t=self.t,
            event_order=1,
            n_iter=1000,
            burning=950,
            particle_idx=particle_idx)
        # Update the triggering kernels
        print(f'[event 1, paricle {particle_idx + 1}]: Updating trigger kernel……')
        particle.update_trigger_kernel(lambda_0_candi_list=lambda_0_candi_list,
                                       beta_candi_list=beta_candi_list,
                                       tau_candi_list=tau_candi_list)

        # Calculate and update the log particle weights
        print(f'[event 1, paricle {particle_idx + 1}]: Updating particle weight……')
        particle.update_paricle_weight(old_particle_weight=self.particle_weight_list[particle_idx], t=self.t, T=self.T,
                                       event_order=1)
        self.particle_weight_list[particle_idx] = particle.new_particle_weight

    def generate_following_event4paritcle(self, particle_idx_pair: (int, Particle), n: int):
        """

        :param n:
        :param particle_idx_pair:
        :return:
        """
        particle_idx = particle_idx_pair[0]
        particle = particle_idx_pair[1]
        print(f'[event {n}, particle {particle_idx + 1}]: Sampling particle……')
        particle.sample_following_event_particle(n)
        lambda_0_candi_list, beta_candi_list, tau_candi_list = particle.sample_hyperparams(
            lambda0_proposal_shape=3,
            lambda0_proposal_scale=2,
            lambda0_prior_shape=2,
            lambda0_prior_scale=2,
            beta_proposal_shape=3,
            beta_proposal_scale=2,
            beta_prior_shape=2,
            beta_prior_scale=2,
            tau_proposal_shape=3,
            tau_proposal_scale=2,
            tau_prior_shape=2,
            tau_prior_scale=2,
            t=self.t,
            event_order=n,
            n_iter=1000,
            burning=950,
            particle_idx=particle_idx)
        print(f'[event {n}, particle {particle_idx + 1}]: Updating trigger kernel……')
        particle.update_trigger_kernel(lambda_0_candi_list=lambda_0_candi_list,
                                       beta_candi_list=beta_candi_list,
                                       tau_candi_list=tau_candi_list)
        print(f'[event {n}, particle {particle_idx + 1}]: Updating particle weight……')
        particle.update_paricle_weight(old_particle_weight=self.particle_weight_list[particle_idx], t=self.t, T=self.T,
                                       event_order=n)
        self.particle_weight_list[particle_idx] = particle.new_particle_weight

    def filtering(self):
        """
        IBHP模型粒子滤波参数估计方法的主要步骤
        :return:
        """
        # 采样第1个event对应的粒子状态，并对每个event的粒子权重进行归一化，判断是否进行重采样
        # 使用线程池并行采样每个粒子
        mg = Manager()
        particle_index_pair_list = [(idx, particle) for idx, particle
                                    in enumerate(self.particle_list)]  # 这个list里的粒子实例会更新
        # 创建进程池能够管理的list类型
        particle_index_pair_list = mg.list(particle_index_pair_list)
        # event 1对应的进程池, 为什么粒子的状态没有变呢。。。
        pool_event_1 = Pool(processes=self.n_particle)
        pool_event_1.map(self.generate_first_event4paricle, particle_index_pair_list)
        pool_event_1.close()
        pool_event_1.join()

        # Normalize the particle weights
        print(f'[event 1]: Normalizing particle weight……')
        self.normalize_particle_weight()
        print(f'[event 1]: Normalized particle weight: \n{self.particle_weight_list}')
        # Resample particles with replacement
        # 计算N_eff（有效粒子数）
        N_eff = 1 / np.sum(np.square(self.particle_weight_list))
        # N_eff小于2/3 * N时重采样
        if N_eff < 2 / 3 * self.n_particle:
            print(f'[event 1]: Resampling particles……')
            self.resample_particles()
        # 输出最好的10个粒子的数据
        for i in np.argsort(- self.particle_weight_list)[: 10]:
            print(f'[event 1, particle {i + 1}] weight: {self.particle_list[i].new_particle_weight}')
            print(f'[event 1, particle {i + 1}] lambda_0: {self.particle_list[i].lambda_0[-1]}')
            print(f'[event 1, particle {i + 1}] beta: {self.particle_list[i].beta}')
            print(f'[event 1, particle {i + 1}] tau: {self.particle_list[i].tau}')

        # 继续采样第2～n个event粒子的状态, 并对每个event的粒子权重进行归一化，判断是否进行重采样
        for n in np.arange(2, self.n_sample + 1):
            # event n对应的线程池
            pool_event_n = Pool(processes=self.n_particle)
            pool_event_n.map(partial(self.generate_following_event4paritcle, n=n), particle_index_pair_list)
            pool_event_n.close()
            pool_event_n.join()

            print(f'[event {n}]: Normalizing particle weight……')
            self.normalize_particle_weight()
            print(f'[event {n}]: Normalized particle weight: {self.particle_weight_list}')
            N_eff = 1 / np.sum(np.square(self.particle_weight_list))
            if N_eff < 2 / 3 * self.n_particle:
                print(f'[event {n}]: Resampling particles……')
                self.resample_particles()
            for i in np.argsort(- self.particle_weight_list)[: 10]:
                print(f'[event {n}, particle {i + 1}] weight: {self.particle_list[i].new_particle_weight}')
                print(f'[event {n}, particle {i + 1}] lambda_0: {self.particle_list[i].lambda_0[-1]}')
                print(f'[event {n}, particle {i + 1}] beta: {self.particle_list[i].beta}')
                print(f'[event {n}, particle {i + 1}] tau: {self.particle_list[i].tau}')


# noinspection SpellCheckingInspection,PyPep8Naming
def main(n_sample=10):
    ibhp = IBHP()
    ibhp.generate_data(n_sample=n_sample)
    print(f'\n{"-" * 40} 生成的观测数据 {"-" * 40}\n')
    print(f'时间戳：{ibhp.timestamp}\n')
    print(f'文本：{transfer_multi_dist_res2vec(ibhp.T)}\n')
    word_corpus = np.arange(100)
    print(f'词典：{word_corpus}\n')
    print(f'\n{"-" * 40} 开始粒子滤波参数估计 {"-" * 40}\n')
    pf = Particle_Filter(t=ibhp.timestamp, T=ibhp.T, n_particle=100, word_corpus=word_corpus, L=3)
    pf.filtering()


if __name__ == '__main__':
    main()
