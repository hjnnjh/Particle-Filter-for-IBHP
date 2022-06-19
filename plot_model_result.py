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
import shutil
from typing import Dict, Tuple

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
                color='darkblue',
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
                color='darkblue',
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
                color='darkblue',
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
    fig.savefig('./img_test/intensity_res.png')
    print('Intensity Plot Save to ./img_test/intensity_res.png')
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
    fig.savefig('./img_test/c_res.png')
    logging.info('C Plot Save to ./img_test/c_res.png')
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
    ax.plot(x, pred_lambda0_array, color='darkblue', label='Pred', alpha=0.5, linewidth=5)
    ax.legend()
    ax.set_xlabel('Event Number')
    ax.set_ylabel(r'$\lambda_0$')
    fig.tight_layout()
    fig.savefig('./img_test/lambda0_res.png')
    logging.info(f'Lambda0 Plot Save to ./img_test/lambda0_res.png')
    plt.close('all')

    # beta
    beta_num = true_beta.shape[1]
    fig, ax = plt.subplots(beta_num, 1, dpi=400, figsize=(26, 5))
    if beta_num == 1:
        ax = [ax]
    for i in range(beta_num):
        ax[i].plot(x, true_beta[:, i], color='r', label='True')
        ax[i].plot(x, pred_beta_array[:, i], color='darkblue', label='Pred', alpha=0.5, linewidth=5)
        ax[i].legend()
        ax[i].set_xlabel('Event Number')
        ax[i].set_ylabel(fr'$\beta_{i}$')
    fig.tight_layout()
    fig.savefig('./img_test/beta_res.png')
    logging.info(f'Beta Plot Save to ./img_test/beta_res.png')
    plt.close('all')

    # tau
    tau_num = true_tau.shape[1]
    fig, ax = plt.subplots(tau_num, 1, dpi=400, figsize=(26, 5))
    if tau_num == 1:
        ax = [ax]
    for i in range(tau_num):
        ax[i].plot(x, true_tau[:, i], color='r', label='True')
        ax[i].plot(x, pred_tau_array[:, i], color='darkblue', label='Pred', alpha=0.5, linewidth=5)
        ax[i].legend()
        ax[i].set_xlabel('Event Number')
        ax[i].set_ylabel(fr'$\tau_{i}$')
    fig.tight_layout()
    fig.savefig('./img_test/tau_res.png')
    logging.info(f'Tau Plot Save to ./img_test/tau_res.png')
    plt.close('all')


def plot_user_intensity(save_dir: str, last_n: int = None, use_timestamp=True):
    """
    plot real data intensity result

    Args:
        save_dir (str): e.g. ./model_result/real_data_tensor_res/A-V-Mani-1
        last_n (int, optional): plot last n results. Defaults to None.
        use_timestamp (bool, optional): use timestamp as xlabels. Defaults to True.
    """
    username = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    intensity_array = np.array([
        torch.load(f'{save_dir}/{filename}/lambda_tn_tensor.pt', map_location='cpu').numpy()
        for filename in os.listdir(f'{save_dir}') if os.path.isdir(f'{save_dir}/{filename}')
    ])
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu').numpy()
    timestamp_array = torch.load(f'{save_dir}/timestamp_tensor.pt', map_location='cpu').numpy()
    fig, ax = plt.subplots(dpi=400, figsize=(20, 5))
    average_pred_intensity_array = np.average(intensity_array, weights=particle_weight, axis=0)
    finished_num = average_pred_intensity_array.shape[0]
    if last_n:
        average_pred_intensity_array = average_pred_intensity_array[-last_n:]
        if use_timestamp:
            ax.plot(timestamp_array[:finished_num][-average_pred_intensity_array.shape[0]:],
                    average_pred_intensity_array,
                    color='darkblue',
                    label='Intensity')
            ax.set_xticks(timestamp_array[:finished_num][-average_pred_intensity_array.shape[0]:])
        else:
            ax.plot(np.arange(average_pred_intensity_array.shape[0]),
                    average_pred_intensity_array,
                    color='darkblue',
                    label='Intensity')
    else:
        if use_timestamp:
            ax.plot(timestamp_array[:finished_num], average_pred_intensity_array, color='darkblue', label='Intensity')
            ax.set_xticks(timestamp_array[:finished_num])
        else:
            ax.plot(np.arange(average_pred_intensity_array.shape[0]),
                    average_pred_intensity_array,
                    color='darkblue',
                    label='Intensity')
    ax.set_xticklabels([])
    ax.set_title(
        f'{intensity_array.shape[0]} Particles, User {username}: Average Intensity, Event Num = {average_pred_intensity_array.shape[0]}'
    )
    ax.set_ylabel(r'$\lambda(t_n)$')
    ax.legend()
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.makedirs(f'./img/{username}')
    if use_timestamp:
        fig.savefig(f'./img/{username}/intensity_with_timestamp.png')
        logging.info(f'{username} intensity plot save to ./img/{username}/intensity_with_timestamp.png')
    else:
        fig.savefig(f'./img/{username}/intensity.png')
        logging.info(f'{username} intensity plot save to ./img/{username}/intensity.png')
    plt.close('all')


def plot_user_hyperparameter(save_dir: str):
    """
    plot real data hyperparameter result

    Args:
        save_dir (str): e.g. ./model_result/real_data_tensor_res/A-V-Mani-1
    """
    username = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    pred_lambda0_array = torch.load(f'{save_dir}/avg_lambda0_tensor.pt', map_location='cpu').numpy()
    pred_beta_array = torch.load(f'{save_dir}/avg_beta_tensor.pt', map_location='cpu').numpy()
    pred_tau_array = torch.load(f'{save_dir}/avg_tau_tensor.pt', map_location='cpu').numpy()
    # print(f'avg lambda0: {np.average(pred_lambda0_array)}')
    # print(f'avg beta: {np.average(pred_beta_array, axis=0)}')
    # print(f'avg tau: {np.average(pred_tau_array, axis=0)}')
    event_num = pred_lambda0_array.shape[0]
    x = np.arange(event_num)

    # plot lambda0
    fig, ax = plt.subplots(dpi=400, figsize=(20, 5))
    ax.plot(x, pred_lambda0_array, color='darkblue', label='Pred')
    ax.set_title(fr'{username}: Average $\lambda_0$')
    ax.set_xlabel('Event Number')
    ax.set_ylabel(fr'$\lambda_0$')
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.makedirs(f'./img/{username}')
    fig.savefig(f'./img/{username}/lambda0.png')
    logging.info(f'{username} lambda0 plot save to ./img/{username}/lambda0.png')
    plt.close('all')

    # plot beta
    beta_dim = pred_beta_array.shape[1]
    fig, ax = plt.subplots(beta_dim, 1, dpi=400, figsize=(20, 5))
    if beta_dim == 1:
        ax = [ax]
    for i in range(beta_dim):
        ax[i].plot(x, pred_beta_array[:, i], color='darkblue', label='Pred')
        ax[i].set_title(fr'{username}: Average $\beta_{i}$')
        ax[i].set_xlabel('Event Number')
        ax[i].set_ylabel(fr'$\beta_{i}$')
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.makedirs(f'./img/{username}')
    fig.savefig(f'./img/{username}/beta.png')
    logging.info(f'{username} beta plot save to ./img/{username}/beta.png')
    plt.close('all')

    # plot tau
    tau_dim = pred_tau_array.shape[1]
    fig, ax = plt.subplots(tau_dim, 1, dpi=400, figsize=(20, 5))
    if tau_dim == 1:
        ax = [ax]
    for i in range(tau_dim):
        ax[i].plot(x, pred_tau_array[:, i], color='darkblue', label='Pred')
        ax[i].set_title(fr'{username}: Average $\tau_{i}$')
        ax[i].set_xlabel('Event Number')
        ax[i].set_ylabel(fr'$\tau_{i}$')
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.makedirs(f'./img/{username}')
    fig.savefig(f'./img/{username}/tau.png')
    logging.info(f'{username} tau plot save to ./img/{username}/tau.png')
    plt.close('all')


def plot_user_c(save_dir: str, color_bar=False):
    """
    plot real data c result

    Args:
        save_dir (str): e.g. ./model_result/real_data_tensor_res/A-V-Mani-1
        color_bar (bool, optional): _description_. Defaults to False.
    """
    username = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    # choose the best particle by particle weight
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu')
    best_particle_index = torch.argmax(particle_weight).item()
    logging.info(f'{username} best particle index: {best_particle_index}')
    c_mat = torch.load(f'{save_dir}/particle-{best_particle_index}/c.pt', map_location='cpu').numpy()

    fig, ax = plt.subplots(dpi=400)
    ms = ax.matshow(c_mat, cmap='Greys')
    ax.set_ylabel('Event (document)')
    ax.set_xlabel('Topic')
    if color_bar:
        fig.colorbar(ms, ax=ax)
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.mkdir(f'./img/{username}')
    fig.savefig(f'./img/{username}/particle_{best_particle_index}_c_mat.png')
    logging.info(
        f'{username} particle-{best_particle_index} c_mat plot save to ./img/{username}/particle_{best_particle_index}_c_mat.png'
    )
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


def plot_user_lambda_k(save_dir: str, lambda_k_mat_window: Tuple[int] = None):
    """
    plot real data lambda_k result

    Args:
        save_dir (str): e.g. ./model_result/real_data_tensor_res/A-V-Mani-1
    """
    username = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu')
    best_particle_index = torch.argmax(particle_weight).item()
    logging.info(f'{username} best particle index: {best_particle_index}')
    particle_path = f'{save_dir}/particle-{best_particle_index}'
    lambda_k_mat = construct_lambda_k_mat(particle_path)
    if lambda_k_mat_window:
        lambda_k_mat = construct_lambda_k_mat(particle_path)
        vertical_lower = np.min(np.argwhere(lambda_k_mat[lambda_k_mat_window[0]] != 0)[:, 0])
        vertical_upper = np.max(np.argwhere(lambda_k_mat[lambda_k_mat_window[1]] != 0)[:, 0])
        lambda_k_mat = lambda_k_mat[lambda_k_mat_window[0]:lambda_k_mat_window[1], vertical_lower:vertical_upper]
    fig, ax = plt.subplots(dpi=400)
    ms = ax.matshow(lambda_k_mat, cmap='GnBu')
    ax.set_ylabel('Event (document)')
    ax.set_xlabel('Topic')
    fig.colorbar(ms, ax=ax)
    fig.tight_layout()
    if not os.path.exists(f'./img/{username}'):
        os.makedirs(f'./img/{username}')
    fig.savefig(f'./img/{username}/particle_{best_particle_index}_lambda_k_mat.png')
    logging.info(
        f'{username} particle-{best_particle_index} lambda_k_mat plot save to ./img/{username}/particle_{best_particle_index}_lambda_k_mat.png'
    )
    plt.close('all')


def plot_factor_word_cloud(save_dir: str, user_word_dict: Dict[str, LabelEncoder], max_words: int = 100):
    """
    plot factor word cloud

    Args:occurrence
        save_dir (str): e.g. ./model_result/real_data_tensor_res/A-V-Mani-1
        user_word_dict (Dict[str, LabelEncoder]): _description_
    """
    user_name = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    # choose the best particle by particle weight
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu')
    best_particle_index = torch.argmax(particle_weight).item()
    particle_path = f'{save_dir}/particle-{best_particle_index}'
    word_classes = user_word_dict[user_name].classes_
    logging.info(f'[{user_name}] word dict length: {len(word_classes)}')
    v_mat = torch.load(f'{particle_path}/v.pt', map_location='cpu')
    c_mat = torch.load(f'{particle_path}/c.pt', map_location='cpu')
    c_mat_count_vert = torch.count_nonzero(c_mat, dim=0)
    wd = WordCloud(background_color='white',
                   width=1920,
                   height=1080,
                   max_words=max_words,
                   font_path='/usr/share/fonts/PingFang-SC-Regular.ttf',
                   relative_scaling=0)
    fig, ax = plt.subplots(dpi=400)
    logging.info(f'plotting {user_name} factor word cloud:')
    if not os.path.exists(f'./img/{user_name}/wordcloud'):
        os.makedirs(f'./img/{user_name}/wordcloud')
    else:
        shutil.rmtree(f'./img/{user_name}/wordcloud')
        os.makedirs(f'./img/{user_name}/wordcloud')
    for idx in torch.arange(c_mat.shape[1]):
        factor_occurrence_time = c_mat_count_vert[idx].item()
        wd_gen = wd.generate_from_frequencies({word_classes[i]: v_mat[i, idx] for i in torch.arange(v_mat.shape[0])})
        ax.imshow(wd_gen, interpolation='bilinear')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(f'./img/{user_name}/wordcloud/factor_{idx}_with_{factor_occurrence_time}_occurrence.png')
        logging.info(
            f'[{user_name}, {idx + 1}/{c_mat.shape[1]}] factor {idx} word cloud save to ./img/{user_name}/wordcloud/factor_{idx}_with_{factor_occurrence_time}_occurrence.png'
        )
        ax.clear()
    logging.info(f'[{user_name}] factor word cloud plot save to ./img/{user_name}/wordcloud')
    plt.close('all')


def get_top_word(save_dir: str, user_word_dict: Dict[str, LabelEncoder], top_word_num: int, save_df: bool = False):
    """
    Get each factor's top words

    Args:
        save_dir (str): _description_
        user_word_dict (Dict[str, LabelEncoder]): _description_
        top_word_num (int): _description_
        save_df (bool): _description_
    """
    user_name = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu')
    best_particle_index = torch.argmax(particle_weight).item()
    particle_path = f'{save_dir}/particle-{best_particle_index}'
    word_classes = user_word_dict[user_name].classes_
    logging.info(f'[{user_name}] word dict length: {len(word_classes)}')
    v_mat = torch.load(f'{particle_path}/v.pt', map_location='cpu')
    c_mat = torch.load(f'{particle_path}/c.pt', map_location='cpu')
    c_mat_count_vert = torch.count_nonzero(c_mat, dim=0)
    top_word = []
    for idx in torch.arange(c_mat.shape[1]):
        temp = []
        temp.append(f'topic {idx + 1}')
        v_vec_idx_descending = v_mat[:, idx].argsort(descending=True)
        temp_top_word = [word_classes[i] for i in v_vec_idx_descending][:top_word_num]
        top_word_str = '; '.join(temp_top_word)
        temp.append(top_word_str)
        occurrence = c_mat_count_vert[idx].item()
        temp.append(occurrence)
        top_word.append(temp)
    top_word_df = pd.DataFrame(top_word, columns=['Topic', 'Top Word', 'Topic Occurrence'])
    if save_df:
        if not os.path.exists(f'./csv_res/{user_name}'):
            os.makedirs(f'./csv_res/{user_name}')
        top_word_df.to_csv(f'./csv_res/{user_name}/top_word.csv', index=False, encoding='utf-8')
        logging.info(f'[{user_name}] top word save to ./csv_res/{user_name}/top_word.csv')


def plot_event_dependency(save_dir: str, event_window: Tuple[int] = None):
    """
    plot event dependency based on c matrix

    Args:
        save_dir (str): _description_
        event_window (Tuple[int]): _description_
    """
    user_name = re.search(r'.*/model_result/.*?/(.*)', save_dir)[1]
    # choose the best particle by particle weight
    particle_weight = torch.load(f'{save_dir}/particle_weight_tensor.pt', map_location='cpu')
    best_particle_index = torch.argmax(particle_weight).item()
    particle_path = f'{save_dir}/particle-{best_particle_index}'
    # load c matrix
    c_mat = torch.load(f'{particle_path}/c.pt', map_location='cpu')
    if event_window:
        c_mat = c_mat[event_window[0]:event_window[1], :]
    dot = graphviz.Digraph(name='event_dependency', format='png', graph_attr={'splines': 'false'})
    event_list = np.array([f'D {i}' for i in torch.arange(c_mat.shape[0])])
    if event_window:
        event_list = np.array([f'D {i}' for i in torch.arange(event_window[0], event_window[1])])
    logging.info(f'[{user_name}] Constructing dependency graph...')
    for idx, event in enumerate(tqdm(event_list)):
        if idx == 0:
            dot.node(event, event)
        else:
            dot.node(event, event)
            for i in torch.arange(idx):
                edge_value = c_mat[idx] @ c_mat[i]
                if edge_value:
                    dot.edge(event_list[i], event_list[idx])
    # logging.info(dot.source)
    dot.render(filename='event_dependency', directory=f'./img/{user_name}/event_dependency')
    logging.info(
        f'[{user_name}] event dependency plot save to ./img/{user_name}/event_dependency/event_dependency.png')


def plot_real_data_res(tensor_read_dir: str, skip_mode=True, default_user_save_dir: str = './img'):
    """
    plot all result of real data

    Args:
        tensor_save_dir (str): e.g. ./model_result/real_data_tensor_res
        skip_mode (bool): if True, skip user
        default_user_save_dir (str): Directory used to save user images, e.g. ./img
    """
    with open('./pickle_files/user_word_corpus_labeled_100.pkl', 'rb') as f:
        user_word_dict = pickle.load(f)
    user_dir = os.listdir(tensor_read_dir)
    user_dir_len = len(user_dir)
    for idx, user in enumerate(user_dir):
        save_dir = f'{tensor_read_dir}/{user}'
        if skip_mode and os.path.exists(default_user_save_dir):
            if user in os.listdir(default_user_save_dir):
                logging.info(f'[{idx + 1}/{user_dir_len}]{user} already exist, skip')
                continue
        logging.info(f'[{idx + 1}/{user_dir_len}] plotting {user}')
        plot_user_intensity(save_dir)
        plot_user_hyperparameter(save_dir)
        plot_user_c(save_dir)
        plot_user_lambda_k(save_dir)
        plot_factor_word_cloud(save_dir, user_word_dict)
    logging.info('all real data result plot done')


def plot_real_agg_data_res(tensor_read_dir: str, skip_mode=True, default_user_save_dir: str = './img'):
    """
    plot all result of real data

    Args:
        tensor_save_dir (str): e.g. ./model_result/real_data_tensor_res
        skip_mode (bool): if True, skip user
        default_user_save_dir (str): Directory used to save user images, e.g. ./img
    """
    user_dir = os.listdir(tensor_read_dir)
    user_dir_len = len(user_dir)
    for idx, user in enumerate(user_dir):
        save_dir = f'{tensor_read_dir}/{user}'
        if skip_mode and os.path.exists(default_user_save_dir):
            if user in os.listdir(default_user_save_dir):
                logging.info(f'[{idx + 1}/{user_dir_len}]{user} already exist, skip')
                continue
        logging.info(f'[{idx + 1}/{user_dir_len}] plotting {user}')
        plot_user_intensity(save_dir)
        plot_user_hyperparameter(save_dir)
        plot_user_c(save_dir)
        plot_user_lambda_k(save_dir)
    logging.info('all real data result plot done')


if __name__ == '__main__':
    plot_event_dependency(save_dir='./model_result/real_data_tensor_res/Aakash-Sarker', event_window=(1, 25))
    plot_user_lambda_k(save_dir='./model_result/real_data_tensor_res/Aakash-Sarker')
    plot_user_c(save_dir='./model_result/real_data_tensor_res/Aakash-Sarker')

    with open('./pickle_files/user_word_corpus_labeled_100.pkl', 'rb') as f:
        user_word_dict = pickle.load(f)
    get_top_word(save_dir='./model_result/real_data_tensor_res/Aakash-Sarker',
                 top_word_num=10,
                 user_word_dict=user_word_dict,
                 save_df=True)

    plot_factor_word_cloud(save_dir='./model_result/real_data_tensor_res/Aakash-Sarker',
                           user_word_dict=user_word_dict,
                           max_words=20)
