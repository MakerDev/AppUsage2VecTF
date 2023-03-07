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
    num_apps  = len(open(os.path.join('data', 'app2id.txt'), 'r').readlines())
    device    = '/GPU:0'

    model = AppUsage2VecTFLite(num_users, num_apps, args.dim, args.seq_length,
                               args.num_layers, args.alpha, args.topk)
    checkpoint_dir = 'checkpoints'

    # TODO: checkpoint_dir should be chosen in consideration of epoch folder.
    latest = tf.train.latest_checkpoint(checkpoint_dir + '/epoch9')
    if latest:
        start_epoch = int(latest.split('/')[-1][-1])
        model.load_weights(latest)
        # tf.saved_model.save(model, f'{checkpoint_dir}/epoch10/test', signatures={
        #     'predict_sample': model.predict_sample.get_concrete_function()
        # })
    else:
        start_epoch = 0

    # data load
    mini = True
    train_dataset = AppUsage2VecDataset(mode='train', mini=mini)
    test_dataset  = AppUsage2VecDataset(mode='test', mini=mini)

    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

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
                users     = tf.convert_to_tensor(data[0])
                time_vecs = tf.convert_to_tensor(data[1])
                app_seqs  = tf.convert_to_tensor(data[2])
                time_seqs = tf.convert_to_tensor(data[3])
                targets   = tf.convert_to_tensor(targets)

                loss      = model.train(users, time_vecs,app_seqs, time_seqs, targets, mode='train')
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
                users     = tf.convert_to_tensor(data[0])
                time_vecs = tf.convert_to_tensor(data[1])
                app_seqs  = tf.convert_to_tensor(data[2])
                time_seqs = tf.convert_to_tensor(data[3])
                targets   = tf.convert_to_tensor(targets)

                scores = model.predict_batch(app_seqs, time_seqs, users, time_vecs, targets)

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
        # tf.saved_model.save(model, f'{checkpoint_dir}/epoch{epoch+1}/saved_model', signatures={
        #     'predict_sample': model.predict_sample.get_concrete_function()
        # })
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
