from nptdms import TdmsFile
import pandas as pd
import os

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

class_name = {0: 'gearbox1', 1: 'gearbox2', 2: 'gearbox3', 3: 'gearbox4', 7: 'gearbox5',
              4: 'gearbox6', 5: 'gearbox7', 6: 'gearbox8', 8: 'gearbox9'}

work_condition = ['_16Hz', '_18Hz', '_20Hz', '_22Hz', '_24Hz', '_26Hz', '_28Hz']

def data_load(root, conditions, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):
    data = []
    label = []
    for lab in class_label:
        name = class_name[lab]
        for condition in conditions:
            num_sample = 0
            path = os.path.join(root, name, name + work_condition[condition] + '.tdms')
            with TdmsFile.open(path) as tdms_file:
                group_name = []
                channel_name = []
                for group in tdms_file.groups():  # TdmsFile可以按组名索引来访问TDMS文件中的组，使用groups()方法直接访问所有组
                    group_name.append(group.name)
                for channel in group.channels():  # TdmsGroup 可以通过通道名称来索引来访问这个组中的一个通道，使用 channels()方法直接访问所有通道
                    channel_name.append(channel.name)
                channel = tdms_file[group_name[0]][channel_name[1]]  # 根据索引读取通道
                all_channel_data = channel[:]  # 将此通道中所有的数据作为numpy数组获取
                data_temp = np.array(all_channel_data)
                data_temp = np.expand_dims(data_temp, axis=1)

            # train
            if flag == 'train':
                start, end = 0, sig_size
                while end <= data_temp.shape[0] and num_sample < num_train_samples:
                    current_sample = data_temp[start:end]
                    current_sample = AddWhiteGaussian(current_sample, SNR)
                    data.append(current_sample)
                    label.append(lab)
                    start += sig_size
                    end += sig_size
                    num_sample += 1

            # val
            elif flag == 'val':
                start, end = sig_size * num_train_samples, sig_size + sig_size * num_train_samples
                while end <= data_temp.shape[0] and num_sample < num_val_samples:
                    current_sample = data_temp[start:end]
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
                while end <= data_temp.shape[0] and num_sample < num_test_samples:
                    current_sample = data_temp[start:end]
                    current_sample = AddWhiteGaussian(current_sample, SNR)
                    data.append(current_sample)
                    label.append(lab)
                    start += sig_size
                    end += sig_size
                    num_sample += 1

    return [data, label]

class THU_data_split(object):
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
        list_train_data = data_load(root=r'D:\datasets\THU',
                                    conditions=self.train_op_conditions,
                                    SNR=self.train_noise_SNR,
                                    num_train_samples=self.num_train_samples,
                                    num_val_samples=self.num_val_samples,
                                    num_test_samples=self.num_test_samples,
                                    sig_size=self.data_length, class_label=self.train_classes, flag='train')

        list_val_data = data_load(root=r'D:\datasets\THU',
                                    conditions=self.val_op_conditions,
                                    SNR=self.val_noise_SNR,
                                    num_train_samples=self.num_train_samples,
                                    num_val_samples=self.num_val_samples,
                                    num_test_samples=self.num_test_samples,
                                    sig_size=self.data_length, class_label=self.val_classes, flag='val')

        list_test_data = data_load(root=r'D:\datasets\THU',
                                   conditions=self.test_op_conditions,
                                   SNR=self.test_noise_SNR,
                                   num_train_samples=self.num_train_samples,
                                   num_val_samples=self.num_val_samples,
                                   num_test_samples=self.num_test_samples,
                                   sig_size=self.data_length, class_label=self.test_classes, flag='test')

        list_ood_test_data = data_load(root=r'D:\datasets\THU',
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