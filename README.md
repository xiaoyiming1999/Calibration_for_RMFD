# Model Calibration Repository for Rotating Machinery Fault Diagnosis from HNU Intelligent Fault Diagnosis Group
### Description of calibration methods for RMFD
This is a benchmark study of trustworthy fault diagnosis based on calibration. We select 7 calibration methods (covering all the four types of calibration methods mentioned in the paper) and a baseline method for the comparative study according to their prevalence, scalability, and applicability. These calibration methods include two regularization methods focal loss and confidece penalty loss, two data augmentation methods label smooth and Mixup, and two uncertainty estimation methods dropblock and deep ensemble. In addition, these methods will be combined with temperature scaling (a post-hoc method) to see if they can benefit from temperature scaling. We analyze in detail the behavior of these calibration methods and derive several valuable conclusions. They can provide insights into subsequent calibration studies and lay a solid foundation for their advancement. Since Our repository is used to process one-dimensional signals, so we made some changes on the model structure and hyperparameters of the methods compared with the original settings. The detailed information on these methods can be found in

|Method                 |URL
|-----------------------|--------------------------------|
|Vanilla                |-                               |
|Temperature scaling    |https://arxiv.org/abs/1706.04599|
|Focal loss             |https://arxiv.org/abs/2002.09437|
|Confidence penalty loss|https://arxiv.org/abs/1701.06548|
|Label smooth           |https://arxiv.org/abs/1906.02629|
|Mixup                  |https://arxiv.org/abs/1710.09412|
|Dropblock              |https://arxiv.org/abs/1906.09551|
|Deep ensemble          |https://arxiv.org/abs/1612.01474|

### Datasets

We use three datasets, including a publicly available dataset and two private datasets. The link to the publicly available dataset PU is
- **[PHM 2009](https://www.phmsociety.org/competition/PHM/09/apparatus)**

The private datasets are the gearbox failure dataset organized by Dr. Yang Cheng from Southeast University and the wind turbine failure dataset organized by Associate Professor Han Te from Beijing Institute of Technology. We are not authorized to share them, but you can contact them.

### Description of key parameters

```
--data_name: str, the dataset used
--model: str, the model used
--method: str, the method used
--num_MC_samplings, int, the number of MC samples for dropblock
--use_temperature, bool, whether use temperature scaling
--data_length, int, the length of a sample
--num_bins, int, the number of bins for reliability diagram

--train_classes, list, classes of training samples
--val_classes, list, classes of validation samples
--test_classes, list, classes of test samples (without OOD classes)
--ood_test, list, classes of test sample (containing OOD classes)

--train_op_conditions, list, operating conditions of training samples
--val_op_conditions, list, operating conditions of validation samples
--test_op_conditions, list, operating conditions of test samples (without OOD classes)

--train_noise_SNR, list, SNR of training samples
--val_noise_SNR, list, SNR of validation samples
--test_noise_SNR, list, SNR of test samples (without OOD classes)
```

### How to use

- download datasets
- Update root in PU.py, SEU.py and THU.py according to the path where you placed the dataset
- use the main.py to test Vanilla, focal loss, confidence penalty loss, Mixup, label smooth, dropblock, and deep ensemble 

- for example, use the following commands to test focal loss in combination with temperature scaling on PU dataset
- `python main.py --data_name PU --use_temperature True --method focal_loss`
- for example, use the following commands to test deep ensemble on SEU dataset that is not combined with temperature scaling
- `python main.py --data_name SEU --use_temperature False --method ensemble`

### Pakages

This repository is organized as:
- [datasets](https://github.com/xiaoyiming1999/Calibration_for_RMFD/tree/main/datasets) contains the data augmentation methods and the Pytorch datasets for time domains.
- [models](https://github.com/xiaoyiming1999/Calibration_for_RMFD/tree/main/models) contains the models and losses used in this project.
- [utils](https://github.com/xiaoyiming1999/Calibration_for_RMFD/tree/main/utils) contains the functions for realization of the training procedure.

### Reference

Part of the code refers to the following open source code:
- [UDTL](https://github.com/ZhaoZhibin/UDTL) from the paper "[Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study](https://ieeexplore.ieee.org/document/9552620)" proposed by Zhao et al.

### Citiation

If this article inspired you, please cite

```
@article{title={Evaluating calibration of deep fault diagnostic models under distribution shift},
        author={Xiao Yiming and Shao Haidong and Liu Bin},
        journal={computers in industry},
        year={2025}}
```

If you have used the code of our repository, please star it, thank you very much.

### Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection, please contact us.

xiaoym@hnu.edu.cn

Mentor E-mail：hdshao@hnu.edu.cn
