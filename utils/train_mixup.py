#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import warnings

import torch
import logging

from torch import optim, nn
from models import resnet18, resnet18_with_dropblock, BiLSTM, CNN_5layer, CNN_7layer, Multi_scale_CNN, compute_entropy
from models import LabelSmoothingCrossEntropy, Confidence, Focal_loss, ECELoss, BS_loss, Temperature_scaling, \
    compute_fpr_at_95tpr
from datasets import PU_data_split, SEU_data_split, THU_data_split
from datasets import Mix_max_process, Mean_std_process, mixup, cutmix
from sklearn.metrics import roc_curve, auc
from utils.reliability_diagram import _reliability_diagram


class TrainMixupUtils(object):
    def __init__(self, args, save_dir):
        self.save_dir = save_dir
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # define model
        if args.model == 'Resnet18':
            self.model = resnet18(out_channel=args.num_classes)
        elif args.model == '7layer_cnn':
            self.model = CNN_7layer(num_cls=args.num_classes)
        elif args.model == '5layer_cnn':
            self.model = CNN_5layer(num_cls=args.num_classes)
        elif args.model == 'multi_scale':
            self.model = Multi_scale_CNN(output_dim=args.num_classes)
        elif args.model == 'BiLSTM':
            self.model = BiLSTM(out_channel=args.num_classes)
        else:
            raise Exception("model not implement")

        # Define the learning parameters
        parameter_list = [{"params": self.model.parameters(), "lr": args.lr}]

        # Define the optimizer
        self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        # Invert the model
        self.model.to(self.device)

        # define the loss
        self.criterion = nn.CrossEntropyLoss()
        self.LS_criterion = LabelSmoothingCrossEntropy(num_cls=args.num_classes)
        self.ece_loss = ECELoss(n_bins=args.num_bins)
        self.focal_loss = Focal_loss(num_class=args.num_classes)
        self.confidence_loss = Confidence()
        self.BS_loss = BS_loss(num_cls=args.num_classes)

    def train(self):

        args = self.args

        # load dataset
        self.datasets = {}
        if args.data_name == 'PU':
            self.datasets['train'], self.datasets['val'], self.datasets['test'], self.datasets['ood'] \
                = PU_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                train_noise_SNR=args.train_noise_SNR,
                val_noise_SNR=args.val_noise_SNR,
                test_noise_SNR=args.test_noise_SNR,
                train_classes=args.train_classes,
                val_classes=args.val_classes,
                test_classes=args.test_classes,
                ood_test_classes=args.ood_test,
                train_op_conditions=args.train_op_conditions,
                val_op_conditions=args.val_op_conditions,
                test_op_conditions=args.test_op_conditions
            ).data_split()
        elif args.data_name == 'SEU':
            self.datasets['train'], self.datasets['val'], self.datasets['test'], self.datasets['ood'] \
                = SEU_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                train_noise_SNR=args.train_noise_SNR,
                val_noise_SNR=args.val_noise_SNR,
                test_noise_SNR=args.test_noise_SNR,
                train_classes=args.train_classes,
                val_classes=args.val_classes,
                test_classes=args.test_classes,
                ood_test_classes=args.ood_test,
                train_op_conditions=args.train_op_conditions,
                val_op_conditions=args.val_op_conditions,
                test_op_conditions=args.test_op_conditions
            ).data_split()
        elif args.data_name == 'THU':
            self.datasets['train'], self.datasets['val'], self.datasets['test'], self.datasets['ood'] \
                = THU_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                train_noise_SNR=args.train_noise_SNR,
                val_noise_SNR=args.val_noise_SNR,
                test_noise_SNR=args.test_noise_SNR,
                train_classes=args.train_classes,
                val_classes=args.val_classes,
                test_classes=args.test_classes,
                ood_test_classes=args.ood_test,
                train_op_conditions=args.train_op_conditions,
                val_op_conditions=args.val_op_conditions,
                test_op_conditions=args.test_op_conditions
            ).data_split()
        else:
            raise Exception("data not implement")

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=False) for x in ['train', 'val', 'test', 'ood']}

        # ----------------------------------------- train and validation ------------------------------------------ #
        for epoch in range(0, args.epoch):
            logging.info('-' * 20 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 20)
            #  learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                all_outputs = []
                all_labels = []
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    if phase == 'train':
                        # data augmentation
                        if args.method == 'MixUp':
                            aug_inputs, index, lam = mixup(inputs)
                        elif args.method == 'CutMix':
                            aug_inputs, index, lam = cutmix(inputs)
                        y_a, y_b = labels, labels[index]

                        # mean-std normalize
                        aug_inputs = Mean_std_process(aug_inputs)
                        aug_inputs = aug_inputs.to(torch.float32)

                    # mean-std normalize
                    inputs = Mean_std_process(inputs)
                    inputs = inputs.to(torch.float32)

                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        outputs = self.model(inputs)
                        # calculate loss function
                        if phase == 'train':
                            aug_outputs = self.model(aug_inputs)
                            loss = lam * self.criterion(aug_outputs, y_a) + \
                                   (1 - lam) * self.criterion(aug_outputs, y_b) + self.criterion(outputs, labels)
                        else:
                            loss = self.criterion(outputs, labels)
                        all_outputs.append(outputs)
                        all_labels.append(labels)

                        # epoch output
                        pred = outputs.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                all_outputs = torch.cat(all_outputs)
                all_labels = torch.cat(all_labels)
                epoch_ece = self.ece_loss(all_outputs, all_labels)
                epoch_ece = epoch_ece.item()
                epoch_NLL = self.criterion(all_outputs, all_labels)
                epoch_NLL = epoch_NLL.item()
                epoch_BS = self.BS_loss(all_outputs, all_labels)
                epoch_BS = epoch_BS.item()

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                logging.info(
                    'Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, {}-ECE: {:.4f}, {}-NLL: {:.4f}, {}-BS: {:.4f}, '
                    'Cost {:.1f} ms'.format(epoch, phase, epoch_loss, phase, epoch_acc, phase, epoch_ece, phase,
                                            epoch_NLL, phase, epoch_BS, 1000 * (time.time() - epoch_start)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # ------------------------------------------------ test ------------------------------------------------- #
        # temperature scaling using validation output
        if args.use_temperature:
            logging.info('-' * 20 + 'After temperature scaling' + '-' * 20)
            # define temperature layer
            self.temperature = Temperature_scaling()
            self.temperature = self.temperature.to(self.device)

            # define optimizer for temperature layer
            parameter_list = [{"params": self.temperature.parameters(), "lr": 0.001}]
            optimizer = optim.LBFGS(parameter_list, lr=0.001, max_iter=1000)

            # train temperature layer using validation dataset
            def closure():
                optimizer.zero_grad()
                calibrated_outputs = self.temperature(all_outputs)
                loss = self.criterion(calibrated_outputs, all_labels)
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            logging.info('-' * 20 + 'before temperature scaling' + '-' * 20)

        # test
        acc = 0.0
        num_test_samples = 0
        all_outputs = []
        all_labels = []
        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['test']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # mean-std normalize
            inputs = Mean_std_process(inputs)
            inputs = inputs.to(torch.float32)

            outputs = self.model(inputs)
            pred = outputs.argmax(dim=1)
            correct = torch.eq(pred, labels).float().sum().item()
            acc += correct
            num_test_samples += labels.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        ece = self.ece_loss(all_outputs, all_labels).item()
        NLL = self.criterion(all_outputs, all_labels).item()
        BS = self.BS_loss(all_outputs, all_labels).item()
        acc = acc / num_test_samples
        logging.info(
            'test-ACC: {:4f}, test-ECE: {:.4f}, test-NLL: {:.4f}, test-BS: {:.4f}'.format(acc, ece, NLL, BS))

        # plot reliability diagram
        labels = all_labels.detach().cpu().numpy()
        labels = labels.reshape(-1)
        softmaxes = all_outputs.softmax(dim=-1)
        confidences, predictions = torch.max(softmaxes, 1)
        confidences = confidences.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        _reliability_diagram(true_labels=labels, pred_labels=predictions,
                             confidences=confidences, num_bins=args.num_bins)

        # ------------------------------------------ OOD detection test ----------------------------------------- #
        all_outputs = []
        all_labels = []
        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['ood']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # mean-std normalize
            inputs = Mean_std_process(inputs)
            inputs = inputs.to(torch.float32)
            outputs = self.model(inputs)

            all_outputs.append(outputs)
            all_labels.append(labels)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_uncertainty = compute_entropy(all_outputs)

        new_labels = torch.where(all_labels > (len(args.train_classes) - 1), 1, 0)
        new_labels = new_labels.detach().cpu().numpy().reshape(-1)
        all_uncertainty = all_uncertainty.detach().cpu().numpy().reshape(-1)
        fpr, tpr, _ = roc_curve(new_labels, all_uncertainty)
        AUC = auc(fpr, tpr)
        logging.info('OOD AUC: {:.4f}'.format(AUC, ))

        fpr = compute_fpr_at_95tpr(new_labels, all_uncertainty)
        logging.info('OOD FPR_at_95TPR: {:.4f}'.format(fpr, ))
