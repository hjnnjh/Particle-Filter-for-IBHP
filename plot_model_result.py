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


# noinspection DuplicatedCode
def plot_intensity(save_dir: str, last_n: int = None, first_n: int = None):
    true_intensity = torch.load(f'{save_dir}/true_lambda_tn.pt', map_location=torch.device('cpu')).numpy()
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location=torch.device('cpu')).numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt', map_location=torch.device('cpu')).numpy()
    pred_intensity_array = np.array(
        [torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt', map_location=torch.device('cpu')).numpy() for filename
         in
         os.listdir(f'{save_dir}')
         if os.path.isdir(f'{save_dir}/{filename}')])
    fig, ax = plt.subplots(dpi=400)
    average_pred_intensity_array = np.average(pred_intensity_array, weights=particle_weight, axis=0)
    if first_n:
        average_pred_intensity_array = average_pred_intensity_array[: first_n]
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]],
                true_intensity[: average_pred_intensity_array.shape[0]], color='r', label='True')
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]], average_pred_intensity_array, color='b',
                label='Pred')
        ax.set_xticks(timestamp_array[: average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'30 Particles, First {first_n} Events Average Intensity',
                     fontsize=10)
    if last_n:
        finished_num = average_pred_intensity_array.shape[0]
        average_pred_intensity_array = average_pred_intensity_array[-last_n:]
        ax.plot(timestamp_array[: finished_num][-average_pred_intensity_array.shape[0]:],
                true_intensity[: finished_num][-average_pred_intensity_array.shape[0]:], color='r', label='True')
        ax.plot(timestamp_array[: finished_num][-average_pred_intensity_array.shape[0]:],
                average_pred_intensity_array, color='b',
                label='Pred')
        ax.set_xticks(timestamp_array[: finished_num][-average_pred_intensity_array.shape[0]:])
        ax.set_xticklabels([])
        ax.set_title(f'30 Particles, Last {last_n} Events Average Intensity',
                     fontsize=10)
    if not first_n and not last_n:
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]],
                true_intensity[: average_pred_intensity_array.shape[0]], color='r', label='True')
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]], average_pred_intensity_array, color='b',
                label='Pred')
        ax.set_xticks(timestamp_array[: average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'30 Particles, Average Intensity')
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_lambda_k():
    lambda_k = np.load('./model_result/model_result_2022_03_28_15_52_04/lambda_k_mat.npy')
    print(lambda_k.shape)


if __name__ == '__main__':
    plot_intensity(save_dir='/Users/huangjinnan/Desktop/PythonProj/Particle-Filter-for-IBHP/'
                            'model_result/model_result_2022_04_16_23_47_30', last_n=150)
