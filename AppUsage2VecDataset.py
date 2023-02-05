import ast
import pandas as pd
from torch.utils.data import Dataset
import torch
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
        time_vector = torch.zeros(31)
        # time_vector = tf.zeros(31)
        
        # time vector one of 7 dim / one of 24 dim
        time_vector[list(map(int, time.split('_')))] = 1
        
        return (torch.LongTensor([user]), time_vector, torch.LongTensor(app_seq), torch.Tensor(time_seq)), torch.LongTensor([target])
        # return (tf.Tensor([user], dtype=tf.int64), 
        #         time_vector,
        #         tf.Tensor(app_seq, dtype=tf.int64),
        #         tf.Tensor(time_seq, dtype=tf.int64),
        #         tf.Tensor([target], dtype=tf.int64))
        
        