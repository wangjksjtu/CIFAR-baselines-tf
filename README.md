# CIFAR-baselines-tf
Baselines for CIFAR10 with tensorflow

## Dependencies
- tensorflow-1.10.0
- numpy, scipy, tabulate, pandas, scipy

__Install the dependencies using `pip`__:
```
pip install requirements.txt
```

## Accuracy
| Model             | Acc.        | Time        | 
| ----------------- |:-----------:|:-----------:|
| [MLP](https://github.com/wangjksjtu/CIFAR-baselines-tf/blob/master/models/mlp.py)     |           |           |
| [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)  |           |           |
| [LeNetV2](https://github.com/wangjksjtu/CIFAR-baselines-tf/blob/master/models/lenet_v2.py)        |           |           |
| [All-CNNs](https://arxiv.org/abs/1412.6806)       |           |           |
| [VGG16](https://arxiv.org/abs/1409.1556)          |           |           |
| [ResNet18](https://arxiv.org/abs/1512.03385)      |           |           |
| [ResNet50](https://arxiv.org/abs/1512.03385)      |           |           |
| [ResNet101](https://arxiv.org/abs/1512.03385)     |           |           |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)      |           |           |
| [DenseNet121](https://arxiv.org/abs/1608.06993)   |           |           |
| [MobileNet](https://arxiv.org/abs/1704.04861)     |           |           |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)   |           |           |
| [ShuffleNet](https://arxiv.org/abs/1707.01083)    |           |           |
| [ShuffleNetV2](https://arxiv.org/abs/1807.11164)  |           |           |



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