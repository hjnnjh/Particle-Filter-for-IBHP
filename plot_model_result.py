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


# noinspection DuplicatedCode
def plot_intensity(save_dir: str, last_n: int = None, first_n: int = None):
    true_intensity = np.load(f'{save_dir}/true_intensity_array.npy')
    particle_weight = np.load(f'{save_dir}/particle_weight.npy')
    timestamp_array = np.load(f'{save_dir}/time_stamp_array.npy')
    pred_intensity_array = np.array(
        [np.load(f'{save_dir}/{filename}/pred_lambda_tn.npy') for filename in
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
        ax.set_title(f'10 Particles, Fix Particle Hyperparameter, First {first_n} Events Average Intensity',
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
        ax.set_title(f'10 Particles, Fix Particle Hyperparameter, Last {last_n} Events Average Intensity',
                     fontsize=10)
    if not first_n and not last_n:
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]],
                true_intensity[: average_pred_intensity_array.shape[0]], color='r', label='True')
        ax.plot(timestamp_array[: average_pred_intensity_array.shape[0]], average_pred_intensity_array, color='b',
                label='Pred')
        ax.set_xticks(timestamp_array[: average_pred_intensity_array.shape[0]])
        ax.set_xticklabels([])
        ax.set_title(f'10 Particles, Fix Particle Hyperparameter, Average Intensity')
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_intensity(save_dir='/Users/huangjinnan/Desktop/PythonProj/Particle-Filter-for-IBHP/model_result'
                            '/model_result_2022_03_28_15_52_04', last_n=100)
