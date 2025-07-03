import os
import pandas as pd

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

def AddWhiteGaussian(seq, SNR):
    Ps = np.sum(seq ** 2) / (seq.shape[0])
    Pn = Ps / (10 ** (SNR / 10))
    # np.random.seed(123)
    noise = np.random.normal(loc=0, scale=1, size=seq.shape)
    noise = noise * np.sqrt(Pn)
    # snr = 10 * np.log10(np.sum(seq ** 2)/np.sum(noise ** 2))
    signal_add_noise = seq + noise
    return signal_add_noise

class_name = {0: 'NORM',
              1: 'GR_broken(断齿)', 2: 'GR_cracking(裂纹)', 3: 'GR_missing(缺齿)', 4: 'GR_pitting(点蚀)',
              5: 'PG_broken(断齿)', 6: 'PG_cracking(裂纹)', 7: 'PG_missing(缺齿)', 8: 'PG_pitting(点蚀)',
              9: 'SG_broken(断齿)', 10: 'SG_cracking(裂纹)', 11: 'SG_missing(缺齿)', 12: 'SG_pitting(点蚀)'}

def data_load(root, conditions, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):
    data = []
    label = []
    for lab in class_label:
        name = class_name[lab]
        class_path = os.path.join(root, name)
        files = os.listdir(class_path)
        all_data_temp = []
        for f in files[1:]:
            real_root = os.path.join(class_path, f)
            data_temp = pd.read_csv(os.path.abspath(real_root), header=None, sep='\t')
            data_temp = np.array(data_temp)
            data_temp = data_temp[0:10 * sig_size, 1]
            all_data_temp.append(data_temp)
        all_data_temp = np.concatenate(all_data_temp)
        all_data_temp = np.expand_dims(all_data_temp, axis=1)

        num_sample = 0
        # train
        if flag == 'train':
            start, end = 0, sig_size
            while end <= all_data_temp.shape[0] and num_sample < num_train_samples:
                current_sample = all_data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1

        # val
        elif flag == 'val':
            start, end = sig_size * num_train_samples, sig_size + sig_size * num_train_samples
            while end <= all_data_temp.shape[0] and num_sample < num_val_samples:
                current_sample = all_data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1

        # test
        elif flag == 'test':
            start, end = sig_size * (num_train_samples + num_val_samples), sig_size + sig_size * (
                    num_train_samples + num_val_samples)
            while end <= all_data_temp.shape[0] and num_sample < num_test_samples:
                current_sample = all_data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1

    return [data, label]

class SEU_data_split(object):
    def __init__(self, data_length,
                 num_train_samples, num_val_samples, num_test_samples,
                 train_noise_SNR, val_noise_SNR, test_noise_SNR,
                 train_classes, val_classes, test_classes, ood_test_classes,
                 train_op_conditions, val_op_conditions, test_op_conditions):
        self.data_length = data_length

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.train_classes = train_classes
        self.val_classes = val_classes
        self.test_classes = test_classes
        self.ood_test_classes = ood_test_classes

        self.train_noise_SNR = train_noise_SNR
        self.val_noise_SNR = val_noise_SNR
        self.test_noise_SNR = test_noise_SNR

        self.train_op_conditions = train_op_conditions
        self.val_op_conditions = val_op_conditions
        self.test_op_conditions = test_op_conditions

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                # AddGaussian(),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Retype(),
                # Scale(1)
            ]),
            'test': Compose([
                Reshape(),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self):
        list_train_data = data_load(root=r'D:\datasets\SEU',
                                    conditions=self.train_op_conditions,
                                    SNR=self.train_noise_SNR,
                                    num_train_samples=self.num_train_samples,
                                    num_val_samples=self.num_val_samples,
                                    num_test_samples=self.num_test_samples,
                                    sig_size=self.data_length, class_label=self.train_classes, flag='train')

        list_val_data = data_load(root=r'D:\datasets\SEU',
                                  conditions=self.val_op_conditions,
                                  SNR=self.val_noise_SNR,
                                  num_train_samples=self.num_train_samples,
                                  num_val_samples=self.num_val_samples,
                                  num_test_samples=self.num_test_samples,
                                  sig_size=self.data_length, class_label=self.val_classes, flag='val')

        list_test_data = data_load(root=r'D:\datasets\SEU',
                                   conditions=self.test_op_conditions,
                                   SNR=self.test_noise_SNR,
                                   num_train_samples=self.num_train_samples,
                                   num_val_samples=self.num_val_samples,
                                   num_test_samples=self.num_test_samples,
                                   sig_size=self.data_length, class_label=self.test_classes, flag='test')

        list_ood_test_data = data_load(root=r'D:\datasets\SEU',
                                       conditions=self.test_op_conditions,
                                       SNR=self.test_noise_SNR,
                                       num_train_samples=self.num_train_samples,
                                       num_val_samples=self.num_val_samples,
                                       num_test_samples=self.num_test_samples,
                                       sig_size=self.data_length, class_label=self.ood_test_classes, flag='test')

        train_data_pd = pd.DataFrame({"data": list_train_data[0], "label": list_train_data[1]})
        val_data_pd = pd.DataFrame({"data": list_val_data[0], "label": list_val_data[1]})
        test_data_pd = pd.DataFrame({"data": list_test_data[0], "label": list_test_data[1]})
        ood_test_data_pd = pd.DataFrame({"data": list_ood_test_data[0], "label": list_ood_test_data[1]})

        train = dataset(list_data=train_data_pd, transform=self.data_transforms['train'])
        val = dataset(list_data=val_data_pd, transform=self.data_transforms['val'])
        test = dataset(list_data=test_data_pd, transform=self.data_transforms['test'])
        ood_test = dataset(list_data=ood_test_data_pd, transform=self.data_transforms['test'])

        return train, val, test, ood_test
