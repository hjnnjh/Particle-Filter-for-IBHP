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
import logging
import os
import pickle
import re
from cmath import log
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from wordcloud import WordCloud

from particle_filter_torch import TENSOR

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'


# noinspection DuplicatedCode
def plot_intensity(save_dir: str, last_n: int = None, first_n: int = None, custom_title=None):
    true_intensity = torch.load(f'{save_dir}/true_lambda_tn.pt', map_location='cpu').numpy()
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu').numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt', map_location='cpu').numpy()
    pred_intensity_array = np.array([
        torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt', map_location='cpu').numpy()
        for filename in os.listdir(f'{save_dir}') if os.path.isdir(f'{save_dir}/{filename}')
    ])
    fig, ax = plt.subplots(dpi=400, figsize=(26, 5))
    average_pred_intensity_array = np.average(pred_intensity_array, weights=particle_weight, axis=0)
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
        ax.set_title(
            f'{particle_weight.shape[0]} Particles, First {first_n} Events Average Intensity, {custom_title}',
            fontsize=10)
    if last_n:
        finished_num = average_pred_intensity_array.shape[0]
        average_pred_intensity_array = average_pred_intensity_array[-last_n:]
        ax.plot(timestamp_array[:finished_num][-average_pred_intensity_array.shape[0]:],
                true_intensity[:finished_num][-average_pred_intensity_array.shape[0]:],
                color='r',
                label='True')
        ax.plot(timestamp_array[:finished_num][-average_pred_intensity_array.shape[0]:],
                average_pred_intensity_array,
                color='b',
                label='Pred',
                alpha=0.5)
        ax.set_xticks(timestamp_array[:finished_num][-average_pred_intensity_array.shape[0]:])
        ax.set_xticklabels([])
        ax.set_title(f'{particle_weight.shape[0]} Particles, Last {last_n} Events Average Intensity, {custom_title}',
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
        ax.set_title(
            f'{particle_weight.shape[0]} Particles, {average_pred_intensity_array.shape[0]} Events, Average Intensity, {custom_title}',
            fontsize=10)
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig('./img/intensity_res.png')
    print('Intensity Plot Save to ./img/intensity_res.png')
    plt.close('all')


def plot_c(simulation_c_path: str, particle_c_path: str):
    simu_c = torch.load(simulation_c_path, map_location='cpu')
    particle_c = torch.load(particle_c_path, map_location='cpu')
    fig, ax = plt.subplots(1, 2, dpi=400)
    ax[0].matshow(simu_c, cmap='YlGnBu')
    ax[1].matshow(particle_c, cmap='YlGnBu')
    ax[0].set_title('Simulation c')
    ax[1].set_title('Particle c')
    fig.tight_layout()
    fig.savefig('./img/c_res.png')
    logging.info('C Plot Save to ./img/c_res.png')
    plt.close('all')


def plot_hyperparameter(save_dir: str, true_lambda0: TENSOR, true_beta: TENSOR, true_tau: TENSOR, last_n: int = None):
    pred_lambda0_array = torch.load(f'{save_dir}/avg_lambda0_tensor.pt', map_location='cpu').numpy()
    pred_beta_array = torch.load(f'{save_dir}/avg_beta_tensor.pt', map_location='cpu').numpy()
    pred_tau_array = torch.load(f'{save_dir}/avg_tau_tensor.pt', map_location='cpu').numpy()
    if last_n:
        pred_lambda0_array = pred_lambda0_array[-last_n:]
        pred_beta_array = pred_beta_array[-last_n:]
        pred_tau_array = pred_tau_array[-last_n:]

    print(f'avg lambda0: {np.average(pred_lambda0_array)}')
    print(f'avg beta: {np.average(pred_beta_array, axis=0)}')
    print(f'avg tau: {np.average(pred_tau_array, axis=0)}')

    event_num = pred_lambda0_array.shape[0]
    true_lambda0 = true_lambda0.repeat(event_num).numpy()
    true_beta = true_beta.repeat(event_num, 1).numpy()
    true_tau = true_tau.repeat(event_num, 1).numpy()

    # lambda0
    fig, ax = plt.subplots(dpi=400, figsize=(26, 5))
    x = np.arange(event_num)
    ax.plot(x, true_lambda0, color='r', label='True')
    ax.plot(x, pred_lambda0_array, color='b', label='Pred', alpha=0.5, linewidth=5)
    ax.legend()
    ax.set_xlabel('Event Number')
    ax.set_ylabel(r'$\lambda_0$')
    fig.tight_layout()
    fig.savefig('./img/lambda0_res.png')
    logging.info(f'Lambda0 Plot Save to ./img/lambda0_res.png')
    plt.close('all')

    # beta
    beta_num = true_beta.shape[1]
    fig, ax = plt.subplots(beta_num, 1, dpi=400, figsize=(26, 5))
    if isinstance(ax, list):
        ax.flatten()
    else:
        ax = [ax]
    for i in range(beta_num):
        ax[i].plot(x, true_beta[:, i], color='r', label='True')
        ax[i].plot(x, pred_beta_array[:, i], color='b', label='Pred', alpha=0.5, linewidth=5)
        ax[i].legend()
        ax[i].set_xlabel('Event Number')
        ax[i].set_ylabel(fr'$\beta_{i}$')
    fig.tight_layout()
    fig.savefig('./img/beta_res.png')
    logging.info(f'Beta Plot Save to ./img/beta_res.png')
    plt.close('all')

    # tau
    tau_num = true_tau.shape[1]
    fig, ax = plt.subplots(tau_num, 1, dpi=400, figsize=(26, 5))
    if isinstance(ax, list):
        ax.flatten()
    else:
        ax = [ax]
    for i in range(tau_num):
        ax[i].plot(x, true_tau[:, i], color='r', label='True')
        ax[i].plot(x, pred_tau_array[:, i], color='b', label='Pred', alpha=0.5, linewidth=5)
        ax[i].legend()
        ax[i].set_xlabel('Event Number')
        ax[i].set_ylabel(fr'$\tau_{i}$')
    fig.tight_layout()
    fig.savefig('./img/tau_res.png')
    logging.info(f'Tau Plot Save to ./img/tau_res.png')
    plt.close('all')


def plot_user_intensity(save_dir: str, username: str):
    intensity_array = np.array([
        torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt', map_location='cpu').numpy()
        for filename in os.listdir(f'{save_dir}') if os.path.isdir(f'{save_dir}/{filename}')
    ])
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu').numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt', map_location='cpu').numpy()
    fig, ax = plt.subplots(dpi=400)
    average_pred_intensity_array = np.average(intensity_array, weights=particle_weight, axis=0)
    ax.plot(timestamp_array[:average_pred_intensity_array.shape[0]],
            average_pred_intensity_array,
            color='b',
            label='Pred',
            alpha=0.5)
    ax.set_xticks(timestamp_array[:average_pred_intensity_array.shape[0]])
    ax.set_xticklabels([])
    ax.set_title(f'10 Particles, {username} Average Intensity, event num = {timestamp_array.shape[0]}')
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.mkdir(f'./img/{username}')
    fig.savefig(f'./img/{username}/{username}_intensity.png')
    logging.info(f'{username} intensity plot save to ./img/{username}/{username}_intensity.png')
    plt.close('all')


def plot_user_hyperparameter(save_dir: str, username: str):
    pred_lambda0_array = torch.load(f'{save_dir}/avg_lambda0_tensor.pt', map_location='cpu').numpy()
    pred_beta_array = torch.load(f'{save_dir}/avg_beta_tensor.pt', map_location='cpu').numpy()
    pred_tau_array = torch.load(f'{save_dir}/avg_tau_tensor.pt', map_location='cpu').numpy()
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
    if not os.path.exists(f'./img/{username}'):
        os.mkdir(f'./img/{username}')
    fig.savefig(f'./img/{username}/{username}_params.png')
    plt.close('all')


def plot_user_c(save_dir: str, username: str):
    c_mat = torch.load(f'{save_dir}/c.pt', map_location='cpu').numpy()
    fig, ax = plt.subplots(dpi=400)
    ms = ax.matshow(c_mat, cmap='YlGnBu')
    ax.set_ylabel('event')
    ax.set_xlabel('factors')
    fig.colorbar(ms, ax=ax)
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.mkdir(f'./img/{username}')
    fig.savefig(f'./img/{username}/{username}_c_mat.png')
    logging.info(f'{username} c_mat save to ./img/{username}/{username}_c_mat.png')
    plt.close('all')


def construct_lambda_k_mat(particle_path: str):
    lambda_k_file_dict = {}
    for pt_filename in os.listdir(particle_path):
        if 'lambda_k' in pt_filename:
            event_str = re.search(r'(.+)(event.+)(\.pt)', pt_filename)[2]
            lambda_k_file_dict[event_str] = torch.load(f'{particle_path}/{pt_filename}', map_location='cpu').numpy()
    sorted_lambda_k_list = sorted(lambda_k_file_dict.items(), key=lambda d: int(re.search(r'event_(\d+)', d[0])[1]))
    v_max_length = sorted_lambda_k_list[-1][1].shape[0]
    new_lambda_k_list = []
    for k, v in sorted_lambda_k_list:
        if v.shape[0] < v_max_length:
            new_v = np.hstack([v, np.zeros(v_max_length - v.shape[0])])
            new_lambda_k_list.append(new_v)
    lambda_k_mat = np.stack(new_lambda_k_list)
    return lambda_k_mat


def plot_user_lambda_k(particle_path: str, username: str):
    lambda_k_mat = construct_lambda_k_mat(particle_path)
    fig, ax = plt.subplots(dpi=400)
    ms = ax.matshow(lambda_k_mat, cmap='YlGnBu')
    ax.set_ylabel('event')
    ax.set_xlabel('factors')
    fig.colorbar(ms, ax=ax)
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.mkdir(f'./img/{username}')
    fig.savefig(f'./img/{username}/{username}_lambda_k_mat.png')
    plt.close('all')


def visualize_factor_word_cloud(v_mat_path: str, user_name: str, particle_path: str,
                                user_word_corpus: Dict[str, LabelEncoder]):
    word_classes = user_word_corpus[user_name].classes_
    print(f'[{user_name}] word corpus: {word_classes}\nlength: {len(word_classes)}')
    v_mat = torch.load(v_mat_path, map_location='cpu').numpy()
    c_mat = torch.load(f'{particle_path}/c.pt', map_location='cpu').numpy()
    c_mat_count_vert = np.count_nonzero(c_mat, axis=0)
    # descending sorted by count numbers
    c_mat_count_vert_descending_index = np.argsort(-c_mat_count_vert)
    wd = WordCloud(background_color='white',
                   width=1920,
                   height=1080,
                   max_words=200,
                   font_path='/usr/share/fonts/PingFang-SC-Regular.ttf',
                   relative_scaling=0)
    fig, ax = plt.subplots(dpi=400)
    print('plotting factor word cloud:')
    for idx in tqdm(c_mat_count_vert_descending_index):
        wd_gen = wd.generate_from_frequencies(
            {word_classes[i]: v_mat[i, idx] * np.power(10, 38)
             for i in np.arange(v_mat.shape[0])})
        ax.imshow(wd_gen, interpolation='bilinear')
        ax.axis('off')
        fig.tight_layout()
        if not os.path.exists(f'./img/{user_name}'):
            os.mkdir(f'./img/{user_name}')
        if not os.path.exists(f'./img/{user_name}/wordcloud'):
            os.mkdir(f'./img/{user_name}/wordcloud')
        fig.savefig(f'./img/{user_name}/wordcloud/'
                    f'{user_name}_wordcloud_factor_occurrence_{c_mat_count_vert[idx]}_descending_factor_{idx}.png')
        ax.clear()
    plt.close('all')

    def plot_real_data_res():
        tensor_res_path = './model_result/real_data_tensor_res'
        pickle_res_path = './model_result/real_data_pickle_res'

        with open('./dataset/pickle_files/user_word_corpus_labeled_100.pkl', 'rb') as f:
            user_word_corpus_labeled_100 = pickle.loads(f.read())

        with open('./dataset/pickle_files/user_event_seq_100.pkl', 'rb') as f:
            user_event_seq_100 = pickle.loads(f.read())

        user_tensor_path_ls = [
            f'{tensor_res_path}/{folder}' for folder in os.listdir(tensor_res_path)
            if os.path.isdir(f'{tensor_res_path}/{folder}')
        ]

        print('plotting model result for each user:')
        for user_tensor_path in tqdm(user_tensor_path_ls):
            user_name = re.search(r'(./model_result/real_data_tensor_res/model_result_)(.+?)(_2022*)',
                                  user_tensor_path)[2]
            # plot_user_c(save_dir=f'{user_tensor_path}/particle-0',
            #             username=user_name)
            # plot_user_lambda_k(save_dir=f'{user_tensor_path}/particle-0',
            #                    username=user_name)
            visualize_factor_word_cloud(v_mat_path=f'{user_tensor_path}/particle-0/v.pt',
                                        user_name=user_name,
                                        particle_path=f'{user_tensor_path}/particle-0',
                                        user_word_corpus=user_word_corpus_labeled_100)


if __name__ == '__main__':
    # visualize_factor_word_cloud(
    #     v_mat_path='./model_result/real_data_tensor_res/model_result_9000-1_2022_05_06_22_14_54/particle-0/v.pt',
    #     user_name='9000-1',
    #     particle_path='./model_result/real_data_tensor_res/model_result_9000-1_2022_05_06_22_14_54/particle-0',
    #     user_word_corpus=user_word_corpus_labeled_100
    # )

    # plot_user_intensity(
    #     save_dir='./model_result/model_result_A-Atwood-4_2022_04_29_11_40_51',
    #     username='A-Atwood-4')

    # plot_user_hyperparameter(
    #     save_dir='./model_result/model_result_A-Atwood-4_2022_04_29_11_40_51',
    #     username='A-Atwood-4')

    # plot_user_c(save_dir='./model_result/test_tensor_result/test_2022_05_25_20_59_14/particle-0', username='test')

    # plot_user_lambda_k(
    #     save_dir=
    #     './model_result/real_data_tensor_res/model_result_A-v-d-Grinten_2022_05_07_00_00_54/particle-0',
    #     username='A-v-d-Grinten')

    plot_hyperparameter(save_dir='./model_result/test_tensor_result/test_2022_05_31_21_55_23',
                        true_lambda0=torch.tensor(2.),
                        true_beta=torch.tensor([3.]),
                        true_tau=torch.tensor([.1]))

    plot_intensity(save_dir='./model_result/test_tensor_result/test_2022_05_31_21_55_23', custom_title='states fixed')

    plot_c(simulation_c_path='model_result/simulation_data/c.pt',
           particle_c_path='model_result/test_tensor_result/test_2022_05_31_21_55_23/particle-50/c.pt')
