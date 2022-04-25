#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# noinspection SpellCheckingInspection
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
import torch
from matplotlib import rcParams

from particle_filter_torch import TENSOR

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'


# noinspection DuplicatedCode
def plot_intensity(save_dir: str, last_n: int = None, first_n: int = None, plot_label=None):
    true_intensity = torch.load(f'{save_dir}/true_lambda_tn.pt', map_location='cpu').numpy()
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu').numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt', map_location='cpu').numpy()
    pred_intensity_array = np.array(
        [torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt', map_location='cpu').numpy() for filename in
         os.listdir(f'{save_dir}')
         if os.path.isdir(f'{save_dir}/{filename}')])
    fig, ax = plt.subplots(dpi=400)
    average_pred_intensity_array = np.average(pred_intensity_array, weights=particle_weight, axis=0)
    if first_n:
        average_pred_intensity_array = average_pred_intensity_array[: first_n]
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]],
                true_intensity[: average_pred_intensity_array.shape[0]], color='r', label='True')
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]], average_pred_intensity_array, color='b',
                label='Pred', alpha=0.5)
        ax.set_xticks(timestamp_array[: average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, First {first_n} Events Average Intensity',
                     fontsize=10)
    if last_n:
        finished_num = average_pred_intensity_array.shape[0]
        average_pred_intensity_array = average_pred_intensity_array[-last_n:]
        ax.plot(timestamp_array[: finished_num][-average_pred_intensity_array.shape[0]:],
                true_intensity[: finished_num][-average_pred_intensity_array.shape[0]:], color='r', label='True')
        ax.plot(timestamp_array[: finished_num][-average_pred_intensity_array.shape[0]:],
                average_pred_intensity_array, color='b',
                label='Pred', alpha=0.5)
        ax.set_xticks(timestamp_array[: finished_num][-average_pred_intensity_array.shape[0]:])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, Last {last_n} Events Average Intensity',
                     fontsize=10)
    if not first_n and not last_n:
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]],
                true_intensity[: average_pred_intensity_array.shape[0]], color='r', label='True')
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]], average_pred_intensity_array, color='b',
                label='Pred', alpha=0.5)
        ax.set_xticks(timestamp_array[: average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, Average Intensity, {plot_label}')
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_hyperparameter(save_dir: str, true_lambda0: TENSOR, true_beta: TENSOR, true_tau: TENSOR):
    pred_lambda0_array = torch.load(f'{save_dir}/avg_lambda0_tensor.pt', map_location='cpu').numpy()
    pred_beta_array = torch.load(f'{save_dir}/avg_beta_tensor.pt', map_location='cpu').numpy()
    pred_tau_array = torch.load(f'{save_dir}/avg_tau_tensor.pt', map_location='cpu').numpy()

    print(f'avg lambda0: {np.average(pred_lambda0_array)}')
    print(f'avg beta: {np.average(pred_beta_array, axis=0)}')
    print(f'avg tau: {np.average(pred_tau_array, axis=0)}')

    event_num = pred_lambda0_array.shape[0]
    true_lambda0 = true_lambda0.repeat(event_num).numpy()
    true_beta = true_beta.repeat(event_num, 1).numpy()
    true_tau = true_tau.repeat(event_num, 1).numpy()
    fig, ax = plt.subplots(3, 3, dpi=400)
    fig.delaxes(ax[2][1])
    fig.delaxes(ax[2][2])
    x = np.arange(event_num)
    for i in np.arange(3):
        ax[0][i].plot(x, true_beta[:, i], color='r')
        ax[0][i].plot(x, pred_beta_array[:, i], color='b', alpha=0.5)
        ax[0][i].set_title(fr'$\beta_{i + 1}$')
        ax[1][i].plot(x, true_tau[:, i], color='r')
        ax[1][i].plot(x, pred_tau_array[:, i], color='b', alpha=0.5)
        ax[1][i].set_title(fr'$\tau_{i + 1}$')
    ax[2][0].plot(x, true_lambda0, color='r')
    ax[2][0].plot(x, pred_lambda0_array, color='b', alpha=0.5)
    ax[2][0].set_title(r'$\lambda_0$')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_intensity(
        save_dir='./model_result/model_result_2022_04_26_00_32_06',
        plot_label='weight is likelihood + prior'
    )
    plot_hyperparameter(
        save_dir='./model_result/model_result_2022_04_26_00_32_06',
        true_lambda0=torch.tensor(2.),
        true_beta=torch.tensor([1., 2., 3.]),
        true_tau=torch.tensor([.3, .2, .1])
    )
