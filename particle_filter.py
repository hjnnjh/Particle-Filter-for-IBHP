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
import logging
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as scipy_gamma_dist

from IBHP_sim import IBHP

# noinspection SpellCheckingInspection
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


# noinspection PyPep8Naming,PyShadowingNames
def transfer_multi_dist_res2vec(T: np.ndarray):
    """
    transform the data matrix generated by multinomial distribution to a concrete word matrix
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


# noinspection PyAttributeOutsideInit,PyMissingConstructor,DuplicatedCode,PyPep8Naming,PyShadowingNames
class Particle(IBHP):
    # noinspection SpellCheckingInspection
    """
        This class implements all the steps of single particle sampling, hyperparameter updating and particle weight
        calculation
        """

    # noinspection PyShadowingNames
    def __init__(self, word_corpus: np.ndarray, L: int = 3):
        """
        :param L: Number of base kernels
        :param word_corpus: dictionary
        """
        self.L = L
        self.word_corpus = word_corpus
        self.S = len(self.word_corpus)
        # self.kappa_all = None

    def update_old_kappa(self, delta_t_vec: np.ndarray):
        """
        update old kappa when generate following events
        :param delta_t_vec: time interval vector: tn-ti
        :return:
        """
        self.kappa = np.vstack((self.kappa, self.kappa[-1, :]))  # add an auxiliary row before update kappa matrix
        assert delta_t_vec.shape[0] == self.kappa.shape[0]
        for i in np.arange(self.kappa.shape[0]):
            base_kernel_func_i = np.vectorize(partial(self.base_kernel_l, delta_t_vec[i]))
            gamma_vec = base_kernel_func_i(self.beta, self.tau).reshape(-1, self.L)
            for k in np.arange(self.kappa.shape[1]):
                self.kappa[i, k] = self.w[:, k] @ gamma_vec[-1, :] * self.c[i, k]

    def sample_first_particle_event_status(self, shape_lambda0, scale_lambda0, shape_beta, scale_beta, shape_tau,
                                           scale_tau):
        """
        Generate the particle state corresponding to the first Event
        :return:
        """
        # 初始化lambda_0, beta_l, tau_l
        self.lambda0 = np.random.gamma(shape_lambda0, scale_lambda0, 1)[0]  # array, shape=(1, )
        self.beta = np.random.gamma(shape_beta, scale_beta, self.L)  # array, shape=(L, )
        self.tau = np.random.gamma(shape_tau, scale_tau, self.L)  # array, shape=(L, )

        self.w_0 = np.array([1 / self.L] * self.L)  # array, shape=(L, )
        self.v_0 = np.array([1 / self.S] * self.S)  # array, shape(S, )

        # 生成K
        self.K = np.random.poisson(self.lambda0, 1)[0]  # K: int
        while self.K == 0:
            logging.info(f'[event 1] 初始化生成的主题数K为0，重新生成')
            self.K = np.random.poisson(self.lambda0, 1)[0]

        # Generate the topic occurrence matrix c, set all c_k=1
        self.c = np.ones((1, self.K))  # matrix, shape=(1, self.K)
        # Generate w
        self.w = np.random.dirichlet(self.w_0, self.K).T  # matrix, shape=(L, K), each column is the weight of each k
        # Generate v
        self.v = np.random.dirichlet(self.v_0, self.K).T  # matrix, shape=(S, K), Each column is a distribution of
        # words for each k

        # calculate kappa
        base_kernel_func = np.vectorize(partial(self.base_kernel_l, 0))
        self.gamma = base_kernel_func(self.beta, self.tau).reshape(-1, self.L)  # shape=(n, L)
        self.kappa = np.zeros((1, self.K))
        for k in np.arange(self.K):
            self.kappa[0, k] = self.w[:, k] @ self.gamma[-1, :] * self.c[0, k]

        # # update kappa_all
        # self.kappa_all = self.kappa.copy().reshape(-1)

    def sample_particle_following_event_status(self, t: np.ndarray, n: int):
        """
        Generate the particle state corresponding to the following event
        :param t: timestamp vector
        :param n: n-th event
        :return:
        """
        if n == 2:
            # 计算lambda_t1，作为self.lambda_tn_arr的第一个元素
            self.lambda_n_minus_1_arr = np.einsum('ij->j', self.kappa / np.count_nonzero(self.kappa, axis=1).
                                                  reshape(self.kappa.shape[0], -1))
            lambda_t1 = np.sum(self.lambda_n_minus_1_arr)
            self.lambda_tn_arr = np.array([lambda_t1])

        # 计算从已有K个主题中生成c的概率
        p = self.lambda_n_minus_1_arr / (self.lambda0 / self.K + self.lambda_n_minus_1_arr)
        # 生成已有K个主题的主题出现情况向量
        generate_old_c_func = np.vectorize(self.generate_c)
        c_old = generate_old_c_func(p)
        # calculate delta_t_vec, tn-ti
        delta_t_vec = t[n - 1] - t[:n]

        # 生成K+
        k_plus = np.random.poisson(self.lambda0 / (self.lambda0 + np.sum(self.lambda_n_minus_1_arr)),
                                   1)[0]
        # 当K+等于0时，检查c_old是不是全为0，如果全为0，则重新生成c_old
        if k_plus == 0:
            while np.all(c_old == 0):
                logging.info(f'[event {n}]当K+为0时，c_old也全为0，重新生成c_old')
                c_old = generate_old_c_func(p)
        # 更新K
        self.K = self.K + k_plus

        if k_plus:
            # 如果K+大于0，初始化新的主题出现情况向量
            c_new = np.ones(k_plus)
            c = np.hstack((c_old, c_new))
            self.c = np.hstack((self.c, np.zeros((self.c.shape[0], k_plus))))  # 将已有的c矩阵用0补全
            self.c = np.vstack((self.c, c))  # 将新的c向量加入c矩阵
            # 更新kappa矩阵中旧的kappa，将时间戳代入代入kappa的计算公式重新计算kappa
            self.update_old_kappa(delta_t_vec)

            # 如果K+大于0, 生成新的w_k, 更新self.w
            w_new = np.random.dirichlet(self.w_0, k_plus).T
            self.w = np.hstack((self.w, w_new))

            # 如果K+大于0, 生成新的v_k, 更新self.v
            v_new = np.random.dirichlet(self.v_0, k_plus).T
            self.v = np.hstack((self.v, v_new))

            # 如果K+大于0, 计算新的kappa
            # kappa_old = self.kappa_all * c_old  # 根据已有的主题出现情况更新old kappa
            kappa_new = np.zeros(k_plus)
            for k in np.arange(k_plus):
                kappa_new[k] = w_new[:, k].T @ self.gamma[-1, :] * c_new[k]

            # update kappa matrix
            kappa = np.hstack((self.kappa[-1, :], kappa_new))
            self.kappa = np.hstack((self.kappa, np.zeros((self.kappa.shape[0], k_plus))))  # 将已有的kappa矩阵用0补全
            self.kappa[-1, :] = kappa  # 将新的kappa加入kappa矩阵
        else:
            self.c = np.vstack((self.c, c_old))
            self.update_old_kappa(delta_t_vec)

        logging.info(f'n={n}时主题出现情况：{self.c}\n')
        logging.info(f'n={n}时的factor kernel：{self.kappa}\n')

        # 计算lambda_tn, 并将lambda_tn加入self.lambda_arr
        non_zero_kappa_idx = np.argwhere(self.kappa[-1, :] != 0)[:, 0]
        lambda_tn_arr = np.einsum('ij->j', self.kappa / np.count_nonzero(self.kappa, axis=1).
                                  reshape(self.kappa.shape[0], -1))
        self.lambda_n_minus_1_arr = lambda_tn_arr
        lambda_tn = np.sum(lambda_tn_arr[non_zero_kappa_idx])
        self.lambda_tn_arr = np.append(self.lambda_tn_arr, lambda_tn)

    def log_hawkes_likelihood(self, t_vec: np.ndarray, n: int, lambda0, beta: np.ndarray, tau: np.ndarray):
        """
        hawkes log likelihood
        :param tau:
        :param beta:
        :param lambda0:
        :param n: 当前的样本顺序，需大于等于1
        :param t_vec: 时间戳向量(观测数据)
        :return:
        """
        # numpy func
        t_div_tau_func = np.vectorize(np.divide, signature='(),(n)->(n)')
        beta_times_tau_func = np.vectorize(lambda x, y: x * y, signature='(n),(n)->(n)')

        # calculate the integral term
        delta_t = t_vec[-1] - t_vec[: n]  # T-ti
        kappa_i_zero_norm_arr = np.count_nonzero(self.kappa, axis=1,
                                                 keepdims=True)  # nonzero element count for each row
        # in kappa matrix
        kappa_neq_zero_index_n = np.argwhere(self.kappa[n - 1, :] != 0)[:, 0]  # nonzero element index for last row
        # in kappa

        exp_term = (1 - np.exp(-t_div_tau_func(delta_t, tau))) / kappa_i_zero_norm_arr
        exp_term = beta_times_tau_func(beta, exp_term)
        whole_exp_term = (self.w[:, kappa_neq_zero_index_n].T @ exp_term.T).T
        whole_exp_term = np.einsum('ij->', whole_exp_term)
        integral_term = -lambda0 * t_vec[-1] - whole_exp_term

        # calculate each lambda_j
        lambda_all = np.zeros(n)
        for j in np.arange(n):  # j range
            kappa_neq_zero_index_j = np.argwhere(self.kappa[j, :] != 0)[:, 0]
            i = np.arange(j + 1)
            tj_minus_ti = t_vec[j] - t_vec[i]  # np.array
            gamma_arr = beta_times_tau_func(beta, np.exp(-t_div_tau_func(tj_minus_ti, tau)))
            numerator = (self.w[:, kappa_neq_zero_index_j].T @ gamma_arr.T).T
            denominator = np.count_nonzero(self.kappa[: j + 1, :], axis=1).reshape(-1, 1)
            lambda_j = np.einsum('ij->', numerator / denominator)  # calculate lambda_j
            lambda_all[j] = lambda_j

        # calculate product for all lambda
        sum_term = np.sum(np.log(lambda_all))

        # calculate likelihood
        log_lh = integral_term + sum_term
        return log_lh

    @staticmethod
    def gamma_dist(shape, scale, x):
        """
        gamma分布的表达式，后续用作计算提议分布和先验分布的p值
        :param shape: parameter
        :param scale: parameter
        :param x: variable
        :return:
        """
        return scipy_gamma_dist.pdf(x=x, a=shape, scale=scale)

    # noinspection PyUnboundLocalVariable,SpellCheckingInspection
    def update_all_hyperparameters(self, N: int, shape_lambda0, scale_lambda0, shape_beta, scale_beta,
                                   shape_tau, scale_tau, t_vec: np.ndarray, n):
        """
        update lambda0, beta, tau
        :param N: sample numbers
        :param shape_lambda0: gamma prior parameter for lambda0
        :param scale_lambda0: gamma prior parameter for lambda0
        :param shape_beta: gamma prior parameter for beta
        :param scale_beta: gamma prior parameter for beta
        :param shape_tau: gamma prior parameter for tau
        :param scale_tau: gamma prior parameter for tau
        :param t_vec: timestamp vector
        :param n: event number
        :return:
        """
        # draw candidate lambda0
        lambda0_candi_arr = np.random.gamma(shape_lambda0, scale_lambda0, N)
        # vectorize func
        lambda0_prior = np.vectorize(partial(self.gamma_dist, shape_lambda0, scale_lambda0))
        # calculate prior for candidate lambda0
        lambda0_p_prior_arr = lambda0_prior(lambda0_candi_arr)

        # draw candidate beta
        beta_candi_mat = np.random.gamma(shape_beta, scale_beta, (N, self.L))
        # vectorize func
        beta_prior = np.vectorize(partial(self.gamma_dist, shape_beta, scale_beta))
        # calculate prior for candidate beta
        beta_p_prior_mat = beta_prior(beta_candi_mat)

        # draw candidate tau
        tau_candi_mat = np.random.gamma(shape_tau, scale_tau, (N, self.L))
        # vectorize func
        tau_prior = np.vectorize(partial(self.gamma_dist, shape_tau, scale_tau))
        # calculate prior for candidate tau
        tau_p_prior_mat = tau_prior(tau_candi_mat)

        # calculate log-likelihood
        log_likelihood = np.vectorize(partial(self.log_hawkes_likelihood, t_vec, n), signature='(),(n),(n)->()')
        log_likelihood_arr = log_likelihood(lambda0_candi_arr, beta_candi_mat, tau_candi_mat)

        # normalize log-likelihood
        log_likelihood_arr = np.exp(log_likelihood_arr - np.max(log_likelihood_arr))
        log_likelihood_arr = log_likelihood_arr / np.sum(log_likelihood_arr)

        # calculate sample weight
        weight_arr = log_likelihood_arr * lambda0_p_prior_arr * np.prod(beta_p_prior_mat, axis=1) * \
                     np.prod(tau_p_prior_mat, axis=1)
        # normalize weight
        weight_arr = weight_arr / np.sum(weight_arr)

        # new hyperparameters
        self.lambda0 = weight_arr @ lambda0_candi_arr
        self.beta = weight_arr @ beta_candi_mat
        self.tau = weight_arr @ tau_candi_mat

    def update_log_particle_weight(self, old_particle_weight, t: np.ndarray, T: np.ndarray, event_order: int):
        """
        calculate and update log particle weight
        :param old_particle_weight:
        :param T: text vector
        :param t: timestamp vector
        :param event_order: The current sample number (sample order), should be greater than or equal to 1
        :return:
        """
        # calculate likelihood for lambda_tn
        kappa_neq_zero_index_n = np.argwhere(self.kappa[event_order - 1, :] != 0)[:, 0]
        i_range = np.arange(event_order)
        tn_minus_ti = t[event_order - 1] - t[i_range]  # array

        t_div_tau_func = np.vectorize(np.divide, signature='(),(n)->(n)')
        beta_times_tau_func = np.vectorize(lambda x, y: x * y, signature='(n),(n)->(n)')

        gamma_arr = beta_times_tau_func(self.beta, np.exp(-t_div_tau_func(tn_minus_ti, self.tau)))
        numerator = (self.w[:, kappa_neq_zero_index_n].T @ gamma_arr.T).T
        denominator = np.count_nonzero(self.kappa[:event_order, :], axis=1).reshape(-1, 1)
        lambda_tn = np.einsum('ij->', numerator / denominator)  # calculate lambda_tn

        # log likelihood for timestamp
        if event_order == 1:
            log_likelihood_timestamp = np.log(lambda_tn) - lambda_tn * t[event_order - 1]
        else:
            log_likelihood_timestamp = np.log(lambda_tn) - lambda_tn * (t[event_order - 1] - t[event_order - 2])

        # log likelihood for text
        vn_avg = np.einsum('ij->i', self.v[:, kappa_neq_zero_index_n]) / np.count_nonzero(
            self.kappa[event_order - 1, :])
        Tn = transfer_multi_dist_res2vec(T)[event_order - 1, :]  # shape=(S, )
        count_dict = Counter(Tn)
        log_likelihood_text = 0
        for k, v in count_dict.items():
            log_likelihood_text += v * np.log(vn_avg[k])
        # Calculate the updated log particle weight
        self.new_log_particle_weight = np.log(old_particle_weight) + log_likelihood_timestamp + log_likelihood_text


# noinspection PyPep8Naming,SpellCheckingInspection,PyAttributeOutsideInit,PyShadowingNames
class Particle_Filter:
    """
    This class controls weight updating, normalization, and resampling of all particles (in parallel)
    """

    def __init__(self, t: np.ndarray, T: np.ndarray, n_particle: int, word_dict: np.ndarray, L: int = 3):
        """
        Generates particles and initializes particle weights

        :param t: 时间戳向量
        :param T: 文本
        :param n_particle: 粒子个数
        :param word_dict: 词典
        :param L: base kernel的个数
        """
        assert len(t) == T.shape[0]
        self.t = t
        self.T = T
        self.n_sample = self.T.shape[0]
        self.n_particle = n_particle
        self.word_corpus = word_dict
        self.particle_list = [Particle(word_corpus=word_dict, L=L) for i in np.arange(self.n_particle)]
        self.particle_weight_arr = np.array([1 / self.n_particle] * self.n_particle)

    def get_particle_list(self):
        return self.particle_list

    def get_particle_num(self):
        return self.n_particle

    def get_partcie_weight_arr(self):
        return self.particle_weight_arr

    def normalize_particle_weight(self):
        """
        Normalize particle weights (map weights to 0-1 interval and sum to 1)
        :return:
        """
        self.particle_weight_arr = np.exp(self.particle_weight_arr - np.max(self.particle_weight_arr))
        self.particle_weight_arr = self.particle_weight_arr / np.sum(self.particle_weight_arr)

    def resample_particles(self):
        """
        Resampling particles, resampling n_particle particles
        :return:
        """
        new_particle_list = []
        sorted_particle_index = np.argsort(self.particle_weight_arr)  # 得到的是粒子权重升序排列的索引
        sorted_particle_weight = self.particle_weight_arr[sorted_particle_index]  # 升序排列后的粒子权重
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
        self.particle_weight_arr = np.array([1 / self.n_particle] * self.n_particle)

    def generate_first_event_status_for_each_paricle(self, particle_idx_pair: Tuple[int, Particle]):
        """
        Generate the state corresponding to the first event for each particle
        :param particle_idx_pair:
        :return:
        """
        particle_idx = particle_idx_pair[0]
        particle = particle_idx_pair[1]
        logging.info(f'[event 1, paricle {particle_idx + 1}] Sampling particle status……')
        particle.sample_first_particle_event_status(shape_lambda0=1, scale_lambda0=1,
                                                    shape_beta=1, scale_beta=1,
                                                    shape_tau=1, scale_tau=1)
        # Update hyperparameters and triggering kernels
        logging.info(f'[event 1, paricle {particle_idx + 1}] Updating hyperparameters and trigger kernel……')
        particle.update_all_hyperparameters(N=10000,
                                            shape_lambda0=1, scale_lambda0=1,
                                            shape_beta=1, scale_beta=1,
                                            shape_tau=1, scale_tau=1,
                                            t_vec=self.t, n=1)

        # Calculate and update the log particle weights
        logging.info(f'[event 1, paricle {particle_idx + 1}] Updating particle weight……')
        particle.update_log_particle_weight(old_particle_weight=self.particle_weight_arr[particle_idx],
                                            t=self.t, T=self.T,
                                            event_order=1)
        return particle_idx, particle

    def generate_following_event_status_for_each_paritcle(self, particle_idx_pair: Tuple[int, Particle], n: int):
        """

        :param n:
        :param particle_idx_pair:
        :return:
        """
        particle_idx = particle_idx_pair[0]
        particle = particle_idx_pair[1]
        logging.info(f'[event {n}, particle {particle_idx + 1}] Sampling particle status……')
        particle.sample_particle_following_event_status(self.t, n)
        logging.info(f'[event {n}, particle {particle_idx + 1}] Updating hyperparameters and trigger kernel……')
        particle.update_all_hyperparameters(N=10000,
                                            shape_lambda0=1, scale_lambda0=1,
                                            shape_beta=1, scale_beta=1,
                                            shape_tau=1, scale_tau=1,
                                            t_vec=self.t, n=n)
        logging.info(f'[event {n}, particle {particle_idx + 1}] Updating particle weight……')
        particle.update_log_particle_weight(old_particle_weight=self.particle_weight_arr[particle_idx],
                                            t=self.t, T=self.T,
                                            event_order=n)
        return particle_idx, particle

    def update_particle_weight_arr(self, particle_index_list: List[Tuple[int, Particle]]):
        """
        更新粒子权重列表
        :param particle_index_list:
        :return:
        """
        for idx, particle in particle_index_list:
            logging.info(f'[particle {idx + 1}] new_particle_weight: {particle.new_log_particle_weight}')
            self.particle_weight_arr[idx] = particle.new_log_particle_weight


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    # Generate test data
    # noinspection SpellCheckingInspection
    ibhp = IBHP()
    n_sample = 100
    ibhp.generate_data(n_sample=n_sample)
    logging.info(f'\n{"-" * 40} Observational data generated {"-" * 40}\n')
    logging.info(f'Timestamp: {ibhp.timestamp}\n')
    logging.info(f'Text: {transfer_multi_dist_res2vec(ibhp.T)}\n')
    word_corpus = np.arange(100)
    logging.info(f'Dictionary: {word_corpus}\n')

    # filtering
    # noinspection PyBroadException,SpellCheckingInspection
    logging.info(f'\n{"-" * 40}  Start particle filter parameter estimation {"-" * 40}\n')
    pf = Particle_Filter(t=ibhp.timestamp, T=ibhp.T, n_particle=10000, word_dict=word_corpus, L=3)
    particle_index_pair_list = [(idx, particle) for idx, particle in enumerate(pf.get_particle_list())]
    # event 1 status
    pool_event_1 = Pool(cpu_count())
    particle_index_pair_list = list(pool_event_1.map(pf.generate_first_event_status_for_each_paricle,
                                                     particle_index_pair_list))
    pool_event_1.close()
    pool_event_1.join()
    pf.update_particle_weight_arr(particle_index_list=particle_index_pair_list)
    pf.normalize_particle_weight()
    logging.info(f'[event 1] Normalized particle weight: \n{pf.get_partcie_weight_arr()}')
    N_eff = 1 / np.sum(np.square(pf.get_partcie_weight_arr()))
    if N_eff < 2 / 3 * pf.get_particle_num():
        logging.info(f'[event 1] Resampling particles……')
        pf.resample_particles()

    # event 2~n status
    for n in np.arange(2, n_sample + 1):
        # process pool for event n
        pool_event_n = Pool(cpu_count())
        particle_index_pair_list = list(
            pool_event_n.map(partial(pf.generate_following_event_status_for_each_paritcle, n=n),
                             particle_index_pair_list))
        pool_event_n.close()
        pool_event_n.join()
        pf.update_particle_weight_arr(particle_index_list=particle_index_pair_list)
        pf.normalize_particle_weight()
        logging.info(f'[event {n}] Normalized particle weight: {pf.get_partcie_weight_arr()}')
        if n == n_sample:
            break
        N_eff = 1 / np.sum(np.square(pf.get_partcie_weight_arr()))
        if N_eff < 2 / 3 * pf.get_particle_num():
            logging.info(f'[event {n}] Resampling particles……')
            pf.resample_particles()

    # Output final result
    # particle weight
    p_weight_arr = pf.get_partcie_weight_arr()
    logging.info(f'所有粒子的权重: {p_weight_arr}\n')

    # Hyperparameters weighted average
    lam_0_arr = np.array([particle.lambda0 for idx, particle in particle_index_pair_list])
    lam_0 = np.average(lam_0_arr, weights=p_weight_arr)
    beta_arr = np.array([particle.beta.reshape(-1) for idx, particle in particle_index_pair_list])
    beta = np.average(beta_arr, axis=0, weights=p_weight_arr)
    tau_arr = np.array([particle.tau.reshape(-1) for idx, particle in particle_index_pair_list])
    tau = np.average(tau_arr, axis=0, weights=p_weight_arr)
    logging.info(f'三个超参数的加权平均值: \n lambda_0: {lam_0}\n beta: {beta}\n tau: {tau}\n')
