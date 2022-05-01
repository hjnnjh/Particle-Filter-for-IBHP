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
import re
from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams

from particle_filter_torch import TENSOR

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'


# noinspection DuplicatedCode
def plot_intensity(save_dir: str,
                   last_n: int = None,
                   first_n: int = None,
                   plot_label=None):
    true_intensity = torch.load(f'{save_dir}/true_lambda_tn.pt',
                                map_location='cpu').numpy()
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt',
                                 map_location='cpu').numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt',
                                 map_location='cpu').numpy()
    pred_intensity_array = np.array([
        torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt',
                   map_location='cpu').numpy()
        for filename in os.listdir(f'{save_dir}')
        if os.path.isdir(f'{save_dir}/{filename}')
    ])
    fig, ax = plt.subplots(dpi=400)
    average_pred_intensity_array = np.average(pred_intensity_array,
                                              weights=particle_weight,
                                              axis=0)
    if first_n:
        average_pred_intensity_array = average_pred_intensity_array[:first_n]
        ax.plot(timestamp_array[:average_pred_intensity_array.shape[0]],
                true_intensity[:average_pred_intensity_array.shape[0]],
                color='r',
                label='True')
        ax.plot(timestamp_array[:average_pred_intensity_array.shape[0]],
                average_pred_intensity_array,
                color='b',
                label='Pred',
                alpha=0.5)
        ax.set_xticks(timestamp_array[:average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, First {first_n} Events Average Intensity',
                     fontsize=10)
    if last_n:
        finished_num = average_pred_intensity_array.shape[0]
        average_pred_intensity_array = average_pred_intensity_array[-last_n:]
        ax.plot(timestamp_array[:finished_num]
                [-average_pred_intensity_array.shape[0]:],
                true_intensity[:finished_num]
                [-average_pred_intensity_array.shape[0]:],
                color='r',
                label='True')
        ax.plot(timestamp_array[:finished_num]
                [-average_pred_intensity_array.shape[0]:],
                average_pred_intensity_array,
                color='b',
                label='Pred',
                alpha=0.5)
        ax.set_xticks(timestamp_array[:finished_num]
                      [-average_pred_intensity_array.shape[0]:])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, Last {last_n} Events Average Intensity',
                     fontsize=10)
    if not first_n and not last_n:
        ax.plot(timestamp_array[:average_pred_intensity_array.shape[0]],
                true_intensity[:average_pred_intensity_array.shape[0]],
                color='r',
                label='True')
        ax.plot(timestamp_array[:average_pred_intensity_array.shape[0]],
                average_pred_intensity_array,
                color='b',
                label='Pred',
                alpha=0.5)
        ax.set_xticks(timestamp_array[:average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, Average Intensity, {plot_label}')
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_hyperparameter(save_dir: str, true_lambda0: TENSOR, true_beta: TENSOR,
                        true_tau: TENSOR, plot_size: int):
    pred_lambda0_array = torch.load(f'{save_dir}/avg_lambda0_tensor.pt',
                                    map_location='cpu').numpy()[plot_size:]
    pred_beta_array = torch.load(f'{save_dir}/avg_beta_tensor.pt',
                                 map_location='cpu').numpy()[plot_size:]
    pred_tau_array = torch.load(f'{save_dir}/avg_tau_tensor.pt',
                                map_location='cpu').numpy()[plot_size:]

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


def plot_user_intensity(save_dir: str, username: str):
    intensity_array = np.array([
        torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt',
                   map_location='cpu').numpy()
        for filename in os.listdir(f'{save_dir}')
        if os.path.isdir(f'{save_dir}/{filename}')
    ])
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt',
                                 map_location='cpu').numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt',
                                 map_location='cpu').numpy()
    fig, ax = plt.subplots(dpi=400)
    average_pred_intensity_array = np.average(intensity_array,
                                              weights=particle_weight,
                                              axis=0)
    ax.plot(timestamp_array[:average_pred_intensity_array.shape[0]],
            average_pred_intensity_array,
            color='b',
            label='Pred',
            alpha=0.5)
    ax.set_xticks(timestamp_array[:average_pred_intensity_array.shape[0]])
    ax.set_xticklabels([])
    ax.set_title(
        f'10 Particles, {username} Average Intensity, event num = {timestamp_array.shape[0]}'
    )
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'./img/{username}_intensity.png')


def plot_user_hyperparameters(save_dir: str, username: str):
    pred_lambda0_array = torch.load(f'{save_dir}/avg_lambda0_tensor.pt',
                                    map_location='cpu').numpy()
    pred_beta_array = torch.load(f'{save_dir}/avg_beta_tensor.pt',
                                 map_location='cpu').numpy()
    pred_tau_array = torch.load(f'{save_dir}/avg_tau_tensor.pt',
                                map_location='cpu').numpy()
    print(f'avg lambda0: {np.average(pred_lambda0_array)}')
    print(f'avg beta: {np.average(pred_beta_array, axis=0)}')
    print(f'avg tau: {np.average(pred_tau_array, axis=0)}')

    event_num = pred_lambda0_array.shape[0]
    fig, ax = plt.subplots(3, 3, dpi=400)
    fig.delaxes(ax[2][1])
    fig.delaxes(ax[2][2])
    x = np.arange(event_num)
    for i in np.arange(3):
        ax[0][i].plot(x, pred_beta_array[:, i], color='b', alpha=0.5)
        ax[0][i].set_title(fr'$\beta_{i + 1}$')
        ax[1][i].plot(x, pred_tau_array[:, i], color='b', alpha=0.5)
        ax[1][i].set_title(fr'$\tau_{i + 1}$')
    ax[2][0].plot(x, pred_lambda0_array, color='b', alpha=0.5)
    ax[2][0].set_title(r'$\lambda_0$')
    fig.tight_layout()
    fig.savefig(f'./img/{username}_params.png')


def plot_user_c(save_dir: str, username: str):
    c_mat = torch.load(f'{save_dir}/c.pt', map_location='cpu').numpy()
    fig, ax = plt.subplots(dpi=400)
    ax.matshow(c_mat)
    ax.set_ylabel('event')
    ax.set_xlabel('factors')
    fig.tight_layout()
    fig.savefig(f'./img/{username}_c_mat.png')


def plot_user_lambda_k(save_dir: str, username: str):
    lambda_k_file_dict = {}
    for pt_filename in os.listdir(save_dir):
        if 'lambda_k' in pt_filename:
            event_str = re.search(r'(.+)(event.+)(\.pt)', pt_filename)[2]
            lambda_k_file_dict[event_str] = torch.load(
                f'{save_dir}/{pt_filename}', map_location='cpu').numpy()
    sorted_lambda_k_list = sorted(
        lambda_k_file_dict.items(),
        key=lambda d: int(re.search(r'event_([0-9]+)', d[0])[1]))
    v_max_length = sorted_lambda_k_list[-1][1].shape[0]
    new_lambda_k_list = []
    for k, v in sorted_lambda_k_list:
        if v.shape[0] < v_max_length:
            new_v = np.hstack([v, np.zeros(v_max_length - v.shape[0])])
            new_lambda_k_list.append(new_v)
    lambda_k_mat = np.stack(new_lambda_k_list)
    fig, ax = plt.subplots(dpi=400)
    ax.matshow(lambda_k_mat)
    ax.set_ylabel('event')
    ax.set_xlabel('factors')
    fig.tight_layout()
    fig.savefig(f'./img/{username}_lambda_k_mat.png')


if __name__ == '__main__':
    # plot_user_intensity(
    #     save_dir='./model_result/model_result_A-Atwood-4_2022_04_29_11_40_51',
    #     username='A-Atwood-4')
    # plot_user_hyperparameters(
    #     save_dir='./model_result/model_result_A-Atwood-4_2022_04_29_11_40_51',
    #     username='A-Atwood-4')
    # plot_user_c(
    #     save_dir=
    #     './model_result/model_result_A-Atwood-4_2022_04_29_11_40_51/particle-0',
    #     username='A-Atwood-4')
    plot_user_lambda_k(
        save_dir=
        './model_result/model_result_A-Atwood-4_2022_04_29_11_40_51/particle-0',
        username='A-Atwood-4')
