# CIFAR-baselines-tf
Baselines for CIFAR10 with [Tensorflow](https://www.tensorflow.org/)

## Dependencies
- tensorflow 1.10.0+
- numpy, scipy, tabulate, pandas

#### Install the dependencies using `pip`:
```
pip install -r requirements.txt
```

## Examples
```
python train.py --model MLP --lr 0.01 --lr_decay 0.1  --scheduler 50 --epoch 100
python train.py --model VGG19 --lr 0.1 --lr_decay 0.1 --scheduler 150,225,300 --epoch 350
```
Note that the __initial learning rate__ for MLP, LeNet and All-CNNs should be set as __0.01__.

## Accuracy
#### No Data Augmentation
| Model             | Accuracy    | Time        | Model             | Accuracy    | Time        |
| ----------------- |:-----------:|:-----------:| ----------------- |:-----------:|:-----------:|
| [MLP](https://github.com/wangjksjtu/CIFAR-baselines-tf/blob/master/models/mlp.py)     | 55.51%    | 0:04:50   | [ResNet18](https://arxiv.org/abs/1512.03385)      | 88.65%    | 4:15:37   |
| [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)  | 63.37%    | 0:09:50   | [ResNet34](https://arxiv.org/abs/1512.03385)      | 88.84%    | 6:58:28    |
| [LeNetV2](https://github.com/wangjksjtu/CIFAR-baselines-tf/blob/master/models/lenet_v2.py)        | 74.46%    | 0:35:36   | [ResNet50](https://arxiv.org/abs/1512.03385)      | 88.55%    | 7:00:00   |
| [All-CNNs](https://arxiv.org/abs/1412.6806)       | 82.06%    | 3:06:17   | [ResNet101](https://arxiv.org/abs/1512.03385)     |           |           |
| [VGG11](https://arxiv.org/abs/1409.1556)          | 86.58%    | 1:45:01   | [ResNet152](https://arxiv.org/abs/1512.03385)     |           |           |
| [VGG13](https://arxiv.org/abs/1409.1556)          | 89.44%    | 2:13:31   | [Wide-ResNet](https://arxiv.org/pdf/1605.07146)   |           |           |
| [VGG16](https://arxiv.org/abs/1409.1556)          | 88.68%    | 10:02:00  | [GoogLeNet](https://arxiv.org/abs/1409.4842)      |           |           |
| [VGG19](https://arxiv.org/abs/1409.1556)          | 88.87%    | 2:44:27   |

#### Data Augmentation
=======
| Model             | Acc.        | Time        | 
| ----------------- |:-----------:|:-----------:|
| [MLP](https://github.com/wangjksjtu/CIFAR-baselines-tf/blob/master/models/mlp.py)     |           |           |
| [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)  |           |           |
| [LeNetV2](https://github.com/wangjksjtu/CIFAR-baselines-tf/blob/master/models/lenet_v2.py)        |           |           |
| [All-CNNs](https://arxiv.org/abs/1412.6806)       |           |           |
| [VGG11](https://arxiv.org/abs/1409.1556)          |           |           |
| [VGG13](https://arxiv.org/abs/1409.1556)          |           |           |
| [VGG16](https://arxiv.org/abs/1409.1556)          |           |           |    
| [VGG19](https://arxiv.org/abs/1409.1556)          |           |           |
| [ResNet18](https://arxiv.org/abs/1512.03385)      |           |           |
| [ResNet34](https://arxiv.org/abs/1512.03385)      |           |           |
| [ResNet50](https://arxiv.org/abs/1512.03385)      |           |           |
| [ResNet101](https://arxiv.org/abs/1512.03385)     |           |           |
| [ResNet152](https://arxiv.org/abs/1512.03385)     |           |           |
| [WideResNet](https://arxiv.org/pdf/1605.07146)    |           |           |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)      |           |           |
>>>>>>> Stashed changes

<!-- TODO
| [DenseNet121](https://arxiv.org/abs/1608.06993)   |           |           |
| [MobileNet](https://arxiv.org/abs/1704.04861)     |           |           |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)   |           |           |
-->

<!-- TODO*2
| [ShuffleNet](https://arxiv.org/abs/1707.01083)    |           |           |
| [ShuffleNetV2](https://arxiv.org/abs/1807.11164)  |           |           |
-->


## Usage
```
Using TensorFlow backend.
usage: train.py [-h] [--dataset DATASET] [--epochs EPOCHS] [--model MODEL]
                [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY]
                [--lr LR] [--lr_decay LR_DECAY] [--scheduler SCHEDULER]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset: mnist/cifar (default: cifar)
  --epochs EPOCHS       Epochs for training models (default: 350)
  --model MODEL         Model architecture (default: VGG16)
  --batch_size BATCH_SIZE
                        Batch size training models (default: 256)
  --weight_decay WEIGHT_DECAY
                        Weight decay (default: 0.0005)
  --lr LR               Initial learning rate (default: 0.1)
  --lr_decay LR_DECAY   Learning rate decay (default: 0.1)
  --scheduler SCHEDULER
                        Learning rate scheduler (default: 150,225,300)
```
