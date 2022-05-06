#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ibhp_quora_dataset.py
@Time    :   2022/5/6 16:40
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import ast
import logging
import os
import pickle
import time
from collections import OrderedDict
from functools import partial
from itertools import chain

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from particle_filter_torch import ParticleFilter

# global vars
device_1 = torch.device('cuda:1')
demo_user_num = 100


def text2occurrence(corpus_length: int, text_index_tensor: torch.Tensor):
    occurrence_tensor = torch.zeros(corpus_length)
    for idx in text_index_tensor:
        occurrence_tensor[idx] += 1
    return occurrence_tensor


# noinspection PyShadowingNames
def construct_user_event_seq(dataset_save_path: str = './dataset/quora_dataset.csv', save_as_pickle=False):
    data = pd.read_csv(dataset_save_path, encoding='utf-8')
    logging.info('Transfer string representation of list to list')
    for col in tqdm(['answer-question title', 'answer-answer text', 'post-post text', 'question-question title']):
        data[col] = data.loc[:, col].map(lambda x: ast.literal_eval(x) if pd.isna(x) is False else x)

    # Construct User Specific Corpus for Each User
    user_word_corpus = OrderedDict()
    logging.info(f'Construct User Specific Corpus for Each User')
    for idx, user in tqdm(enumerate(data['username'].unique())):
        df_user = data[data['username'] == user]
        user_corpus_list = []
        for col in ['answer-question title', 'answer-answer text', 'question-question title', 'post-post text']:
            non_nan_series = df_user[col].map(lambda x: True if isinstance(x, list) else False)
            series_user_col_non_nan = df_user.loc[non_nan_series, col]
            user_col_word_corpus_ls = list(chain.from_iterable(series_user_col_non_nan.values))
            user_corpus_list.extend(user_col_word_corpus_ls)
        user_word_corpus[user] = user_corpus_list
        if idx == demo_user_num - 1:
            break
    # remove duplicated word in corpus
    for k, v in user_word_corpus.items():
        user_word_corpus[k] = list(set(v))
    # corpus with `LabelEncoder`
    user_word_corpus_labeled = OrderedDict()  # value is fitted `LabelEncoder` instance
    logging.info(f'Encoding user corpus')
    for k, v in tqdm(user_word_corpus.items()):
        le = LabelEncoder()
        le.fit(v)
        user_word_corpus_labeled[k] = le
    # need to save user
    pickle_save_path = './dataset/pickle_files'
    if save_as_pickle:
        if not os.path.exists(pickle_save_path):
            os.mkdir(pickle_save_path)
        with open(f'{pickle_save_path}/user_word_corpus_labeled_{demo_user_num}.pkl', 'wb') as f:
            f.write(pickle.dumps(user_word_corpus_labeled))
        logging.info('Encoded user corpus saved')

    # Construct Event Sequence for Each User
    data_processed = data.copy()
    # transfer `nan` to `[]`
    for col in ['answer-question title', 'answer-answer text', 'question-question title', 'post-post text']:
        data_processed.loc[data_processed[col].isnull(), col] = data_processed.loc[
            data_processed[col].isnull(), col].apply(lambda x: [])

    data_processed['text'] = data_processed['answer-question title'] + data_processed['answer-answer text'] + \
                             data_processed['question-question title'] + data_processed['post-post text']

    user_event_seq = OrderedDict()
    logging.info('Construct Event Sequence for Each User\n')
    for idx, (user, le) in tqdm(enumerate(user_word_corpus_labeled.items())):
        user_text_tensor_series = data_processed.loc[data_processed['username'] == user, 'text'].map(
            lambda text_ls: torch.from_numpy(le.transform(text_ls)) if isinstance(text_ls,
                                                                                  list) and text_ls else text_ls)
        user_timestamp = data_processed.loc[data_processed['username'] == user, 'timedelta_to_hours'].values
        user_event_seq[user] = {
            'time_seq': torch.from_numpy(user_timestamp).to(torch.float32),
            'text_tensor_seq': user_text_tensor_series.values.tolist()
        }
        if idx == demo_user_num - 1:
            break
    # delete empty text `[]` and corresponding timestamp
    logging.info('Delete empty text `[]` and corresponding timestamp')
    for user, seq in tqdm(user_event_seq.items()):
        empty_text_index_ls = [i for i, x in enumerate(seq['text_tensor_seq']) if x == []]
        all_text_index_ls = list(range(len(seq['text_tensor_seq'])))
        non_empty_index_ls = list(filter(lambda idx: idx not in empty_text_index_ls, all_text_index_ls))
        if empty_text_index_ls:
            for idx in sorted(empty_text_index_ls, reverse=True):
                del seq['text_tensor_seq'][idx]
            seq['time_seq'] = seq['time_seq'][non_empty_index_ls]

        # transfer text to 2-d text occurrence tensor
        user_corpus_tensor = torch.from_numpy(
            user_word_corpus_labeled[user].transform(user_word_corpus_labeled[user].classes_)).to(
            torch.int32)  # just like corpus in simulation program
        seq['text_tensor_seq'] = list(
            map(partial(text2occurrence, user_corpus_tensor.shape[0]), seq['text_tensor_seq']))
        seq['text_tensor_seq'] = torch.stack(seq['text_tensor_seq'])
    if save_as_pickle:
        with open(f'{pickle_save_path}/user_event_seq_{demo_user_num}.pkl', 'wb') as f:
            f.write(pickle.dumps(user_event_seq))
        logging.info('User event dict saved')
    return user_event_seq, user_word_corpus_labeled


# noinspection PyShadowingNames
def load_pickle_data(pickle_save_path: str = './dataset/pickle_files'):
    with open(f'{pickle_save_path}/user_word_corpus_labeled_100.pkl', 'rb') as f:
        user_word_corpus_labeled_100 = pickle.loads(f.read())
    with open(f'{pickle_save_path}/user_event_seq_100.pkl', 'rb') as f:
        user_event_seq_100 = pickle.loads(f.read())
    return user_word_corpus_labeled_100, user_event_seq_100


# noinspection PyShadowingNames
def ibhp_fit(user_event_seq, user_word_corpus_labeled, device: torch.device = device_1, save_res=False,
             save_as_pickle=False):
    for user, seq in user_event_seq.items():
        assert seq['text_tensor_seq'].shape[0] == seq['time_seq'].shape[0]
        user_corpus_tensor = torch.from_numpy(
            user_word_corpus_labeled[user].transform(user_word_corpus_labeled[user].classes_)).to(torch.int32)
        pf = ParticleFilter(
            n_sample=seq['time_seq'].shape[0],
            n_particle=10,
            word_corpus=user_corpus_tensor,
            timestamp_tensor=seq['time_seq'],
            text_tensor=seq['text_tensor_seq'],
            lambda0=torch.tensor(4.),
            beta=torch.tensor([3., 4., 5.]),
            tau=torch.tensor([.3, .4, .5]),
            alpha_lambda0=torch.tensor(4.),
            alpha_beta=torch.tensor(4.),
            alpha_tau=torch.tensor(1.5),
            random_num=2500,
            device=device
        )
        pf.filtering(username=user, save_res=save_res)
        if save_as_pickle:
            user_res_pickle_path = './model_result/real_data_pickle_res'
            if not os.path.exists(user_res_pickle_path):
                os.mkdir(user_res_pickle_path)
            with open(f'{user_res_pickle_path}/pf_{user}.pkl', 'wb') as f:
                f.write(pickle.dumps(pf))
            for particle in pf.particle_list:
                if not os.path.exists(f'{user_res_pickle_path}/{user}_particle_pickle'):
                    os.mkdir(f'{user_res_pickle_path}/{user}_particle_pickle')
                with open(f'{user_res_pickle_path}/{user}_particle_pickle/particle_{user}_{particle.particle_idx}.pkl',
                          'wb') as f:
                    f.write(pickle.dumps(particle))
            logging.info(f'{user} pickle result saved')
            time.sleep(2)


if __name__ == "__main__":
    # user_event_seq, user_word_corpus_labeled = construct_user_event_seq(save_as_pickle=True)
    # ibhp_fit(user_event_seq, user_word_corpus_labeled)
    user_word_corpus_labeled_100, user_event_seq_100 = load_pickle_data()
    ibhp_fit(
        user_event_seq=user_event_seq_100,
        user_word_corpus_labeled=user_word_corpus_labeled_100,
        save_res=True,
        save_as_pickle=True,
        device=device_1
    )
