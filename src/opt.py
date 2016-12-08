import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--img_dir', type=str, default="data/LSP/images/")
    parser.add_argument('--train_csv_fn', type=str, default="data/LSP/train_joints.csv")
    parser.add_argument('--test_csv_fn', type=str, default="data/LSP/test_joints.csv")
    parser.add_argument('--result_dir', type=str, default="result/LSP/1/")
    parser.add_argument('--n_joints', type=int, default=16)
    parser.add_argument('--inputRes', type=int, default=256)
    parser.add_argument('--outputRes', type=int, default=64)
    parser.add_argument('--scale', type=int, default=0.25)
    parser.add_argument('--rotate', type=int, default=30)
    

    args = parser.parse_args()

    return args
