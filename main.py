#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import logging
import warnings
from utils.logger import setlogger
from utils.train_regularization import TrainRegularizationUtils
from utils.train_dropblock import TrainDropBlockUtils
from utils.train_ensemble import TrainEnsembleUtils
from utils.train_mixup import TrainMixupUtils

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model parameters
    parser.add_argument('--model_name', type=str, default='Model Calibration')
    parser.add_argument('--model', type=str, default='Resnet18',
                        choices=['Resnet18', '7layer_cnn', 'Resnet18withDropblock', 'BiLSTM', 'multi_scale',
                                 '5layer_cnn'])
    parser.add_argument('--method', type=str, default='Vanilla',
                        choices=['Vanilla', 'label_smooth', 'focal_loss', 'confidence', 'MixUp', 'dropblock',
                                 'ensemble', 'CutMix'])
    parser.add_argument('--num_MC_samplings', type=int, default=16)
    parser.add_argument('--use_temperature', type=bool, default=True)

    # data parameters
    parser.add_argument('--data_name', type=str, default='PU', choices=['PU', 'SEU', 'THU'])
    parser.add_argument('--data_length', type=int, default=1024)
    parser.add_argument('--in_channel', type=int, default=1)

    parser.add_argument('--num_train_samples', type=int, default=400)
    parser.add_argument('--train_classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--train_op_conditions', type=list, default=[2])
    parser.add_argument('--train_noise_SNR', type=int, default=5)

    parser.add_argument('--num_val_samples', type=int, default=50)
    parser.add_argument('--val_classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--val_op_conditions', type=list, default=[2])
    parser.add_argument('--val_noise_SNR', type=int, default=5)

    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--test_classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--ood_test', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument('--test_op_conditions', type=list, default=[2])
    parser.add_argument('--test_noise_SNR', type=int, default=5)

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)

    # optimization information
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--steps', type=str, default='15,25')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--num_bins', type=int, default=10)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    if args.method == 'Vanilla' or args.method == 'label_smooth' \
            or args.method == 'confidence' or args.method == 'focal_loss':
        trainer = TrainRegularizationUtils(args, save_dir)
        trainer.setup()
        trainer.train()

    elif args.method == 'dropblock':
        trainer = TrainDropBlockUtils(args, save_dir)
        trainer.setup()
        trainer.train()

    elif args.method == 'ensemble':
        trainer = TrainEnsembleUtils(args, save_dir)
        trainer.setup()
        trainer.train()

    elif args.method == 'MixUp' or args.method == 'CutMix':
        trainer = TrainMixupUtils(args, save_dir)
        trainer.setup()
        trainer.train()





