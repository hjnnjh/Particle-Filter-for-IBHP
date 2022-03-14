#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   plot_model_result.py
@Time    :   2022/3/11 4:12 PM
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from particle_filter import plot_parameters


def plot_intensity():
    true_intensity = np.load('./model-result/true_intensity_array.npy')
    particle_weight = np.load('./model-result/particle_weight.npy')
    pred_intensity_array = np.array(
        [np.load(f'./model-result/{filename}/pred_lambda_tn.npy') for filename in os.listdir('./model-result')
         if os.path.isdir(f'./model-result/{filename}')])
    average_pred_intensity_array = np.average(pred_intensity_array, weights=particle_weight, axis=0)
    fig, ax = plt.subplots(dpi=400)
    x_true = np.arange(1, true_intensity.shape[0] + 1)
    x_pred = np.arange(1, average_pred_intensity_array.shape[0] + 1)
    ax.plot(x_true, true_intensity, color='r')
    ax.plot(x_pred, average_pred_intensity_array, color='b')
    ax.set_xlabel('n')
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.set_title('Intensity')
    fig.tight_layout()
    plt.show()


def read_file_plot_parameters():
    pred_lambda_0 = np.load(
        './model-result/pred_lambda_0.npy')
    pred_beta = np.load('./model-result/pred_beta.npy')
    pred_tau = np.load('./model-result/pred_tau.npy')
    plot_parameters(true_lambda_0=2, true_beta=np.array([1, 2, 3]), true_tau=np.array([0.3, 0.2, 0.1]),
                    pred_lambda_0=pred_lambda_0, pred_beta=pred_beta, pred_tau=pred_tau,
                    n_sample=pred_lambda_0.shape[0])


if __name__ == '__main__':
    read_file_plot_parameters()
