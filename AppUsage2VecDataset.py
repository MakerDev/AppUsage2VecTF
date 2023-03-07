import ast
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf


class AppUsage2VecDataset(Dataset):
    """AppUsage2Vec Dataset
    
    Args:
        mode(str): which dataset will you make, 'train' or 'test'
    """

    def __init__(self, mode, mini=True):
        if mode == 'train':
            self.df = pd.read_csv(f'data/train{"_mini" if mini else ""}.txt', sep='\t')
        else:
            self.df = pd.read_csv(f'data/test{"_mini" if mini else ""}.txt', sep='\t')

        self.df['app_seq'] = self.df['app_seq'].apply(ast.literal_eval)
        self.df['time_seq'] = self.df['time_seq'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df.iloc[idx]['user']
        time = self.df.iloc[idx]['time']
        target = self.df.iloc[idx]['app']
        app_seq = self.df.iloc[idx]['app_seq']
        time_seq = self.df.iloc[idx]['time_seq']
        time_vector = np.zeros(31)
        # time vector one of 7 dim / one of 24 dim
        time_vector[list(map(int, time.split('_')))] = 1

        return (np.asarray([user], dtype=np.int64),
                np.asarray(time_vector, dtype=np.float32),
                np.asarray(app_seq, dtype=np.int64),
                np.asarray(time_seq, dtype=np.float32)), np.asarray([target], dtype=np.int64)

    @staticmethod
    def generate_tf_dataset(batch_size, shuffle=True, mini=True, mode='train'):
        if mode == 'train':
            df = pd.read_csv(f'data/train{"_mini" if mini else ""}.txt', sep='\t')
        else:
            df = pd.read_csv(f'data/test{"_mini" if mini else ""}.txt', sep='\t')

        df['app_seq'] = df['app_seq'].apply(ast.literal_eval)
        df['time_seq'] = df['time_seq'].apply(ast.literal_eval)

        users_np = df['user'].to_numpy(dtype=np.int64).reshape((len(df), 1))
        targets_np = df['app'].to_numpy(dtype=np.int64).reshape((len(df), 1))
        time = df[:]['time']
        time_vectors_np = np.zeros((len(df), 31), dtype=np.float32)
        for i, t in enumerate(time):
            time_vectors_np[i][list(map(int, t.split('_')))] = 1

        app_seqs = df['app_seq']
        time_seqs = df['time_seq']
        
        app_seqs_np = np.zeros((len(df), 4), dtype=np.int64)
        time_seqs_np = np.zeros((len(df), 4), dtype=np.float32)
        for i in range(len(df)):
            app_seqs_np[i] = np.asarray(app_seqs[i], dtype=np.int64)
            time_seqs_np[i] = np.asarray(time_seqs[i], dtype=np.float32)

        dataset_dict = {
            "user": users_np,
            "time_vector": time_vectors_np,
            "app_seq": app_seqs_np,
            "time_seq": time_seqs_np,
            "target": targets_np
        }

        for key, value in dataset_dict.items():
            dataset_dict[key] = tf.convert_to_tensor(value)

        dataset = tf.data.Dataset.from_tensor_slices(dataset_dict).batch(batch_size, drop_remainder=True)

        if shuffle:
            dataset = dataset.shuffle(len(df))

        return dataset
