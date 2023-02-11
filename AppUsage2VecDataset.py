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
        return  len(self.df)
    
    def __getitem__(self, idx):
        user = self.df.iloc[idx]['user']
        time = self.df.iloc[idx]['time']
        target = self.df.iloc[idx]['app']
        app_seq = self.df.iloc[idx]['app_seq']
        time_seq = self.df.iloc[idx]['time_seq']
        # time_vector = torch.zeros(31)
        time_vector = np.zeros(31)
        import torch
        # time vector one of 7 dim / one of 24 dim
        time_vector[list(map(int, time.split('_')))] = 1
        
        # return (tf.convert_to_tensor([user], dtype=tf.int64), 
        #         tf.convert_to_tensor(time_vector, dtype=tf.float32), 
        #         tf.convert_to_tensor(app_seq, dtype=tf.int64), 
        #         tf.convert_to_tensor(time_seq, dtype=tf.float32)), tf.convert_to_tensor([target], dtype=tf.int64)
        return (np.asarray([user], dtype=np.int64), 
                np.asarray(time_vector, dtype=np.float32), 
                np.asarray(app_seq, dtype=np.int64), 
                np.asarray(time_seq, dtype=np.float32)), np.asarray([target], dtype=np.int64)
        # return (torch.LongTensor([user]), time_vector, torch.LongTensor(app_seq), torch.Tensor(time_seq)), torch.LongTensor([target])
        