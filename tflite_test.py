import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from AppUsage2VecDataset import AppUsage2VecDataset
from AppUsage2VecTFLite import AppUsage2VecTFLite
import os
import utils
import torch


def run_test(model: AppUsage2VecTFLite, dataloader):
    corrects = [0, 0, 0]
    for _, (data, targets) in enumerate(dataloader):
        with tf.device('/GPU:0'):
            users     = tf.convert_to_tensor(data[0])
            time_vecs = tf.convert_to_tensor(data[1])
            app_seqs  = tf.convert_to_tensor(data[2])
            time_seqs = tf.convert_to_tensor(data[3])
            targets   = tf.convert_to_tensor(targets)

            scores = model.predict_batch(app_seqs, time_seqs, users, time_vecs)

        for idx, k in enumerate([1, 5, 10]):
            correct = torch.sum(torch.eq(torch.topk(torch.Tensor(scores.numpy()), dim=1, k=k).indices, 
                                            torch.Tensor(targets.numpy()))).item()
            corrects[idx] += correct

    accs = [x/len(dataset) for x in corrects]
    print("[EVALUATION] Acc: {:.5f} / {:.5f} / {:.5f}".format(accs[0], accs[1], accs[2]))


def run_tflite_test(interpreter:tf.lite.Interpreter, dataloader):
    interpreter.allocate_tensors()

    corrects = [0, 0, 0]
    for _, (data, targets) in enumerate(dataloader):
        with tf.device('/GPU:0'):            
            users     = data[0]
            time_vecs = data[1]
            app_seqs  = data[2]
            time_seqs = data[3]
            targets   = targets
        
            # Get input and output tensors.
            input_details  = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Test the model on random input data.
            # 2, 3, 0, 1 순서로 넣어야함.
            interpreter.set_tensor(input_details[0]['index'], app_seqs)
            interpreter.set_tensor(input_details[1]['index'], time_seqs)
            interpreter.set_tensor(input_details[2]['index'], users)
            interpreter.set_tensor(input_details[3]['index'], time_vecs)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            

        for idx, k in enumerate([1, 5, 10]):
            correct = torch.sum(torch.eq(torch.topk(torch.Tensor(np.asarray(output_data)), dim=1, k=k).indices, 
                                            torch.Tensor(np.asarray(targets)))).item()
            corrects[idx] += correct

    accs = [x/len(dataset) for x in corrects]
    print("[EVALUATION] Acc: {:.5f} / {:.5f} / {:.5f}".format(accs[0], accs[1], accs[2]))


if __name__ == '__main__':
    args      = utils.parse_args()
    num_users = len(open(os.path.join('data', 'user2id.txt'), 'r').readlines())
    num_apps  = len(open(os.path.join('data', 'app2id.txt'), 'r').readlines())

    model = AppUsage2VecTFLite(num_users, num_apps, args.dim, args.seq_length,
                            args.num_layers, args.alpha, args.topk)

    latest_model_dir = tf.train.latest_checkpoint('checkpoints/epoch8')
    model.load_weights(latest_model_dir)

    mini    = True
    dataset = AppUsage2VecDataset(mode='test', mini=mini)

    # Run origianl TF model
    run_test(model=model, dataloader=DataLoader(dataset, batch_size=64))
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="tflite_converted_model/model_ep8.tflite")
    run_tflite_test(interpreter=interpreter, dataloader=DataLoader(dataset, batch_size=1))
