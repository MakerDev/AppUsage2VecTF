import os
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from AppUsage2VecDataset import AppUsage2VecDataset
from AppUsage2Vec import AppUsage2Vec
from tensorflow.python.client import device_lib
keras = tf.keras
from keras.optimizers import Adam
import keras.metrics as metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pytorch implementation of AppUsage2Vec model")

    parser.add_argument('--epoch', type=int, default=15,
                        help="The number of epochs")
    parser.add_argument('--batch_size', type=int,
                        default=128, help="The size of batch")
    parser.add_argument('--dim', type=int, default=64,
                        help="The embedding size of users and apps")
    parser.add_argument('--seq_length', type=int, default=4,
                        help="The length of previously used app sequence")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Number of layers in DNN")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="Discount coefficient for loss function")
    parser.add_argument('--topk', type=float, default=5,
                        help="Topk for loss function")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # random seed
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # data load
    mini = False
    train_dataset = AppUsage2VecDataset(mode='train', mini=mini)
    test_dataset  = AppUsage2VecDataset(mode='test', mini=mini)

    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    num_users = len(open(os.path.join('data', 'user2id.txt'), 'r').readlines())
    num_apps  = len(open(os.path.join('data', 'app2id.txt'), 'r').readlines())
    device    = '/GPU:0'
    
    model = AppUsage2Vec(num_users, num_apps, args.dim, args.seq_length,
                         args.num_layers, args.alpha, args.topk, device)
    checkpoint_dir = 'checkpoints'

    #TODO: checkpoint_dir should be chosen in consideration of epoch folder.
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        start_epoch = int(latest.split('/')[-1][-1])
        model.load_weights(latest)
    else:
        start_epoch = 0

    optimizer = Adam(learning_rate=args.lr)

    # train & evaluation
    train_loss  = metrics.Mean()
    itr         = 1
    p_itr       = 500
    Ks          = [1, 5, 10]
    acc_history = [[0, 0, 0]]
    best_acc    = 0
    for epoch in range(start_epoch, args.epoch):
        for _, (data, targets) in tqdm(enumerate(train_loader)):
            with tf.device(device), tf.GradientTape() as tape:
                users     = tf.convert_to_tensor(data[0].numpy())
                time_vecs = tf.convert_to_tensor(data[1].numpy())
                app_seqs  = tf.convert_to_tensor(data[2].numpy())
                time_seqs = tf.convert_to_tensor(data[3].numpy())
                targets   = tf.convert_to_tensor(targets.numpy())

                loss = model(users, time_vecs, app_seqs, time_seqs, targets, mode='train')
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            if itr % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(epoch+1, itr, train_loss.result()/p_itr))
                train_loss.reset_states()
            itr += 1

        corrects = [0, 0, 0]
        for _, (data, targets) in enumerate(test_loader):
            with tf.device(device):
                users     = tf.convert_to_tensor(data[0].numpy())
                time_vecs = tf.convert_to_tensor(data[1].numpy())
                app_seqs  = tf.convert_to_tensor(data[2].numpy())
                time_seqs = tf.convert_to_tensor(data[3].numpy())
                targets   = tf.convert_to_tensor(targets.numpy())

                scores = model(users, time_vecs, app_seqs, time_seqs, targets)

            for idx, k in enumerate(Ks):
                correct = torch.sum(torch.eq(torch.topk(torch.Tensor(scores.numpy()), dim=1, k=k).indices, 
                                             torch.Tensor(targets.numpy()))).item()
                corrects[idx] += correct
        
        accs = [x/len(test_dataset) for x in corrects]
        acc_history.append(accs)
        print("[EVALUATION] Epoch: {} - Acc: {:.5f} / {:.5f} / {:.5f}".format(epoch + 1, accs[0], accs[1], accs[2]))

        if accs[2] > best_acc:
            best_acc = accs[2]

        accs      = [x*100 for x in accs]
        save_path = f'{checkpoint_dir}/epoch{epoch+1}'
        os.makedirs(save_path, exist_ok=True)
        model.save_weights(f'{checkpoint_dir}/epoch{epoch+1}/{accs[0]:.1f}_{accs[1]:.1f}_{accs[2]:.1f}')

    print("BEST ACC@10: {}".format(best_acc))

    # visualization
    fig = plt.figure()
    acc_history = list(map(list, zip(*acc_history)))
    for idx, k in enumerate(Ks):
        plt.plot(acc_history[idx], label='acc@{}'.format(k))
    plt.xlabel('epoch')
    plt.ylabel('accuracy@k')
    plt.legend(loc='upper left')
    plt.xticks(np.arange(args.epoch+1))
    plt.ylim(0, 1)
    fig.savefig('result.png')

if __name__ == "__main__":
    main()
