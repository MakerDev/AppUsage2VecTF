import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
# from tqdm import tqdm
from AppUsage2VecDataset import AppUsage2VecDataset
from AppUsage2VecTFLite import AppUsage2VecTFLite

keras = tf.keras
from keras.optimizers import Adam
import keras.metrics as metrics
import utils


def main():
    args = utils.parse_args()

    # random seed
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    num_users = len(open(os.path.join('data', 'user2id.txt'), 'r').readlines())
    num_apps = len(open(os.path.join('data', 'app2id.txt'), 'r').readlines())
    device = '/GPU:0'

    model = AppUsage2VecTFLite(num_users, num_apps, args.dim, args.seq_length,
                               args.num_layers, args.alpha, args.topk)
    checkpoint_dir = 'checkpoints'

    start_epoch = 0
    # TODO: checkpoint_dir should be chosen in consideration of epoch folder.
    # latest = tf.train.latest_checkpoint(checkpoint_dir + '/epoch9')
    # if latest:
    #     # start_epoch = int(latest.split('/')[-1][-1])
    #     model.load_weights(latest)

    # data load
    mini = True
    train_dataset = AppUsage2VecDataset.generate_tf_dataset(batch_size=args.batch_size, mini=mini, mode='train')
    test_dataset = AppUsage2VecDataset.generate_tf_dataset(batch_size=args.batch_size, mini=mini, mode='test',
                                                           shuffle=False)
    test_size = len(test_dataset) * args.batch_size

    optimizer = Adam(learning_rate=args.lr)

    # train & evaluation
    train_loss = metrics.Mean()
    itr = 1
    p_itr = 5
    Ks = [1, 5, 10]
    acc_history = [[0, 0, 0]]
    best_acc = 0

    for epoch in range(start_epoch, args.epoch):
        for data_dict in tqdm(train_dataset):
            with tf.device(device), tf.GradientTape() as tape:
                users = data_dict['user']
                time_vecs = data_dict['time_vector']
                app_seqs = data_dict['app_seq']
                time_seqs = data_dict['time_seq']
                targets = data_dict['target']

                loss = model(users, time_vecs, app_seqs, time_seqs, targets, mode='train')
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            if itr % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(epoch + 1, itr, train_loss.result() / p_itr))
                train_loss.reset_states()
            itr += 1

        corrects = [0, 0, 0]
        for _, data_dict in enumerate(test_dataset):
            with tf.device(device):
                users = data_dict['user']
                time_vecs = data_dict['time_vector']
                app_seqs = data_dict['app_seq']
                time_seqs = data_dict['time_seq']
                targets = data_dict['target']

                scores = model.predict_batch(app_seqs, time_seqs, users, time_vecs)

            for idx, k in enumerate(Ks):
                correct = torch.sum(torch.eq(torch.topk(torch.Tensor(scores.numpy()), dim=1, k=k).indices,
                                             torch.Tensor(targets.numpy()))).item()
                corrects[idx] += correct

        accs = [x / test_size for x in corrects]
        acc_history.append(accs)
        print("[EVALUATION] Epoch: {} - Acc: {:.5f} / {:.5f} / {:.5f}".format(epoch + 1, accs[0], accs[1], accs[2]))

        if accs[2] > best_acc:
            best_acc = accs[2]

        accs = [x * 100 for x in accs]
        save_path = f'{checkpoint_dir}/epoch{epoch + 1}'
        os.makedirs(save_path, exist_ok=True)
        model.save_weights(f'{checkpoint_dir}/epoch{epoch + 1}/{accs[0]:.1f}_{accs[1]:.1f}_{accs[2]:.1f}')

    print("BEST ACC@10: {}".format(best_acc))

    # visualization
    fig = plt.figure()
    acc_history = list(map(list, zip(*acc_history)))
    for idx, k in enumerate(Ks):
        plt.plot(acc_history[idx], label='acc@{}'.format(k))
    plt.xlabel('epoch')
    plt.ylabel('accuracy@k')
    plt.legend(loc='upper left')
    plt.xticks(np.arange(args.epoch + 1))
    plt.ylim(0, 1)
    fig.savefig('result.png')
    plt.show()


if __name__ == "__main__":
    main()
