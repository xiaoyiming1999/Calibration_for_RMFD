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

We use three datasets, including a publicly available dataset and two private datasets.
The link to the publicly available dataset PU is
- **[PHM 2009](https://www.phmsociety.org/competition/PHM/09/apparatus)**

The private datasets are the gearbox failure dataset organized by Dr. Yang Cheng from Southeast University and the wind turbine failure dataset organized by Associate Professor Han Te from Beijing Institute of Technology. We are not authorized to share them, but you can contact them.
