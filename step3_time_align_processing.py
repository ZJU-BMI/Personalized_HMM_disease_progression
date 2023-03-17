import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import os
import pickle
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys



warnings.filterwarnings('ignore')
df = pd.read_csv('step2_merge_data_original.csv')
def import_longitudinal_data(max_seq_len=8, normalize=True, normal_mode='normal'):
    df = pd.read_csv('step2_merge_data_original.csv')
    grouped = df.groupby(['subject_id'])
    features = df.iloc[:, 5:]
    id_list = pd.unique(df['subject_id'])  #Screen out the patient's id number
    N = len(id_list)    #Statistical follow-up times of patients
    x_dim = features.shape[1]   
    data_x = np.zeros([N, max_seq_len, x_dim])   #Nnumber of patients，max_seq_len：longest time，x_dim：Number of clinical features
    time = np.zeros([N], dtype=np.int32)    # time label
    label = np.zeros([N, max_seq_len], dtype=np.str_)   #label
    DX_bl = np.zeros([N, max_seq_len], dtype=np.str_)  # label
    seq_lens = np.zeros([N], dtype=np.int32)
    time_seq = -np.ones([N, max_seq_len], dtype=np.int32)
    ids = np.zeros([N], dtype=np.int32)

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)
        len_x = tmp.shape[0]
        ids[i] = tmp_id
        if len_x > max_seq_len:
            drop_len = len_x - max_seq_len
            idx = np.arange(1, len_x - 1)
            drop_idx = np.random.choice(idx, drop_len, replace=False)
            tmp.drop(index=drop_idx, inplace=True)

        time_seq[i, :len_x] = tmp['time'][0:len_x]
        data_x[i, :len_x, :] = tmp.iloc[:len_x, 5:]
        label[i, :len_x] = tmp['label'][0:len_x]
        DX_bl[i, :len_x] = tmp['DX_bl'][0:len_x]
        seq_lens[i] = len_x
        # time to progression
        # time[i] = np.max(tmp['time'])
        #
        # # indicator of progression
        # label[i] = tmp['label'][len_x]
        # number of x

    def get_attn_mask(seq_lens, max_len=100):
        N = seq_lens.shape[0]
        attn_mask = np.zeros([N, max_len])
        for i in range(N):
            attn_mask[i, :seq_lens[i]] = 1
        return attn_mask
    x_dim = data_x.shape[2]
    attn_mask = get_attn_mask(seq_lens, max_len=max_seq_len)

    return (x_dim, max_seq_len), data_x, (time, DX_bl,label,attn_mask, time_seq, ids)


def longitudinal_data_to_csv(ids, data_x, time, DX_bl,label,feature_names):
    N = len(ids)
    seq_len, x_dim = data_x.shape[1], data_x.shape[2]
    colnames = ['subject_id', 'time','DX_bl','label'] + feature_names
    df = pd.DataFrame(columns=colnames, index=np.repeat(np.arange(N), seq_len))
    for i, idx in enumerate(ids):
        df.iloc[i * seq_len: i * seq_len + seq_len, 0] = idx
        df.iloc[i * seq_len: i * seq_len + seq_len, 1] = time[i, :]
        df.iloc[i * seq_len: i * seq_len + seq_len, 2] = DX_bl[i, :]
        df.iloc[i * seq_len: i * seq_len + seq_len, 3] = label[i, :]
        df.iloc[i * seq_len: i * seq_len + seq_len, 4:] = data_x[i, :, :]

    return df

if __name__ == '__main__':
    (x_dim, max_seq_len), data_x, (time, DX_bl,label, attn_mask, time_seq, ids) = import_longitudinal_data()
    df = pd.read_csv('step2_merge_data_original.csv')
    feature_names = list(df.iloc[:, 5:].columns)
    df = longitudinal_data_to_csv(ids, data_x, time_seq, DX_bl, label, feature_names)
    df.to_csv('mergedata_time_align.csv', encoding='gbk')

