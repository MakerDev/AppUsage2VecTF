import argparse


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
