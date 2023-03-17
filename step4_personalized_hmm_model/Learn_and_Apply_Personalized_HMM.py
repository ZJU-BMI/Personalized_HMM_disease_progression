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

sys.setrecursionlimit(10000)
from piohmm import HMM
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle as pickle

warnings.filterwarnings('ignore')
path = " "
def import_longitudinal_data(max_seq_len=8, normalize=True, normal_mode='normal'):
    df = pd.read_csv('step3_mergedata_time_align.csv')
    #if normalize:
    #    df.iloc[:, 6:] = get_normalization(np.asarray(df.iloc[:, 6:]), normal_mode)
    grouped = df.groupby(['subject_id'])
    features = df.iloc[:, 6:]
    id_list = pd.unique(df['subject_id'])
    N = len(id_list)
    x_dim = features.shape[1]
    data_x = np.zeros([N, max_seq_len, x_dim])
    time = np.zeros([N], dtype=np.int32)
    label = np.zeros([N, max_seq_len], dtype=np.str_)
    seq_lens = np.zeros([N], dtype=np.int32)
    time_seq = np.zeros([N, max_seq_len], dtype=np.int32)
    np.random.seed(1234)
    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)
        len_x = tmp.shape[0]
        if len_x > 8:
            drop_len = len_x - max_seq_len
            idx = np.arange(1, len_x - 1)
            drop_idx = np.random.choice(idx, drop_len, replace=False)
            tmp.drop(index=drop_idx, inplace=True)

        time_seq[i, :len_x] = tmp['time'][0:len_x]
        data_x[i, :len_x, :] = tmp.iloc[:len_x, 6:]
        label[i, :len_x] = tmp['label'][0:len_x]
        seq_lens[i] = len_x
        # time to progression
        # time[i] = np.max(tmp['time'])
        #
        # # indicator of progression
        # label[i] = tmp['label'][len_x]
        # number of x

    x_dim = data_x.shape[2]
    #attn_mask = get_attn_mask(seq_lens, max_len=max_seq_len)

    return (x_dim, max_seq_len), data_x, (time, label, time_seq)

if __name__ == '__main__':
    (x_dim, max_seq_len), data_x, (time, label, time_seq) = import_longitudinal_data()


#  5-fold cross-validation
len_data_x=data_x.shape[0]
idx=np.arange(start=0, stop=len_data_x, step=1, dtype=np.int)
print(idx)

k = 5
# Shuffle idx randomly
np.random.shuffle(idx)
idx_per_k=[]
len_per_k = len_data_x // k
for i in range(k - 1):
    idx_per_k.append(idx[i * len_per_k : (i + 1) * len_per_k])
idx_per_k.append(idx[(k - 1) * len_per_k :])
print(len(idx_per_k))
for i in range(k):
    print(len(idx_per_k[i]))

data_x_per_k = []
for i in range(k):
    data_x_per_k.append(data_x[idx_per_k[i]])
print(data_x_per_k[i])
train_data1=data_x_per_k[0]
train_data2=data_x_per_k[1]
train_data3=data_x_per_k[2]
train_data4=data_x_per_k[3]
train_data5=data_x_per_k[4]

train_dataset1 = np.vstack([train_data1, train_data2, train_data3, train_data4])
test_dataset1 = train_data5
train_dataset2 = np.vstack([train_data2, train_data3, train_data4, train_data5])
test_dataset2 = train_data1
train_dataset3 = np.vstack([train_data1, train_data3, train_data4, train_data5])
test_dataset3 = train_data2
train_dataset4 = np.vstack([train_data1, train_data2, train_data4, train_data5])
test_dataset4 = train_data3
train_dataset5 = np.vstack([train_data1, train_data2, train_data3, train_data5])
test_dataset5 = train_data4


K=3
#  Take first-fold cross-validation as an example

remove_idx = np.where(np.sum(~np.isnan(train_dataset1[:, :, 0]), axis=1) == 1)
train_1 = np.delete(train_dataset1, remove_idx, 0)
# get time and observation masks
n, t, d, = train_1.shape
time_mask = np.ones((n, t))
for i in range(n):
    train_1[i, :, 0].shape
    ind = np.where(~np.isnan(train_1[i, :, 0]))[0][-1] + 1
    time_mask[i, ind:] = 0
missing_mask = (~np.isnan(train_1[:, :, 0]))
train_1[np.isnan(train_1)] = 0
# convert everything to tensors
X_train_1 = torch.Tensor(train_1)
TM_train = torch.Tensor(time_mask)
OM_train = torch.Tensor(missing_mask)

piohmm = HMM(X_train_1, k=K, OM=OM_train,TM=TM_train,full_cov=False, priorV=False, io=False, personalized=True, personalized_io=False,
                state_io=False, UT=False, device='cuda:2', eps=1e-18, priorMu=True, var_fill=0.5)
piohmm_params, _, _, elbo, b_hat, _ = piohmm.learn_model(num_iter=2200, intermediate_save=False)
# calculate the most probable sequence
mps_train = piohmm.predict_sequence(piohmm_params, n_sample=b_hat)
piohmm_train1 = {'piohmm_params': piohmm_params, 'mps': mps_train, 'elbo': elbo, 'b_hat': b_hat}
save_path = os.path.join(path, 'train/K=3')
if not os.path.exists(save_path):
    os.makedirs(save_path)
with open(os.path.join(save_path, 'model_tarin1.pkl'), 'wb') as handle:
    pickle.dump(piohmm_train1, handle)

remove_idx = np.where(np.sum(~np.isnan(test_dataset1[:, :, 0]), axis=1) == 1)
validation_1 = np.delete(test_dataset1, remove_idx, 0)
# get time and observation masks
n, t, d, = validation_1.shape
time_mask = np.ones((n, t))
for i in range(n):
    validation_1[i, :, 0].shape
    ind = np.where(~np.isnan(validation_1[i, :, 0]))[0][-1] + 1
    time_mask[i, ind:] = 0
missing_mask = (~np.isnan(validation_1[:, :, 0]))
validation_1[np.isnan(validation_1)] = 0
# convert everything to tensors
X_val_1 = torch.Tensor(validation_1)
TM_test = torch.Tensor(time_mask)
OM_test = torch.Tensor(missing_mask)

model.change_data(X_val_1, OM=OM_test,TM=TM_test,full_cov=False, reset_VI=True,params=params_hat)
piohmm_params, _, _, elbo, b_hat, _ = model.learn_vi_params(piohmm_params, num_iter=2200)
mps_test = piohmm.predict_sequence(piohmm_params, n_sample=b_hat)
piohmm_test1 = {'piohmm_params': piohmm_params, 'mps': mps_test, 'elbo': elbo, 'b_hat': b_hat}
save_path = os.path.join(path, 'validation/K=3')
if not os.path.exists(save_path):
    os.makedirs(save_path)
with open(os.path.join(save_path, 'model_validation1.pkl'), 'wb') as handle:
    pickle.dump(piohmm_test1, handle)

