# Images Classification with CNN

使用卷积神经网络进行图片分类。

## 环境

- Python 3.0以上
- Pytorch
- numpy
- scikit-learn
- scipy

## 数据集

使用 CIFAR-10 数据集的图片进行训练。

所有的图片共分为10类，训练数据中每个类中包含5000张图片(下载链接：http://www.cs.toronto.edu/~kriz/cifar.html)

类别如下：

```
飞机 汽车 鸟类 猫类 鹿类 狗类 青蛙 马 船 卡车
```

## CNN卷积神经网络

### CNN模型

```python
# coding: utf-8
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # 父类构造方法
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input image size: [3, 32, 32]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),            # [3,32,32]->[64,32,32]
            nn.BatchNorm2d(64),                   
            nn.ReLU(), 
            nn.MaxPool2d(2, 2, 0),                # [64,32,32]->[64,16,16]

            nn.Conv2d(64, 128, 3, 1, 1),          #[64,16,16]->[128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                #[128,16,16]->[128,8,8]

            nn.Conv2d(128, 256, 3, 1, 1),         #[128,8,8]->[256,8,8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                #[256,8,8]->[256,4,4]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 10]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x
```

### 训练与验证

运行 `run_cnn.py`文件，可以开始训练。

训练20个epoches结果后，准确度在测试集上已经到达78%，训练好的模型存在文件 `cnn_para.mdl` 中

```
...
Epoch: 51
[ Train | 051/080 ] loss = 0.05214, acc = 0.98147
[ Test | 051/080 ] loss = 1.47089, acc = 0.73981
Epoch: 52
[ Train | 052/080 ] loss = 0.05445, acc = 0.98102
[ Test | 002/080 ] loss = 1.22657, acc = 0.78273
Epoch: 53
[ Train | 053/080 ] loss = 0.04184, acc = 0.98612
[ Test | 053/080 ] loss = 1.32570, acc = 0.76879
...
```



## ResNet 残差网络

用18 层的深度残差网络 ResNet18进行CIFAR10 图片数据集上训练与测试。

### ResNet模型

标准的 ResNet18 接受输入为224 × 224 大小的图片数据，这里我们将 ResNet18 的第一个卷积层进行适量调整，使得它输入大小为32 × 32，输出维度为 10。调整后的 ResNet18 网络结构如下：

```python
# ResNet18 
        self.in_channels = 64
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = [7,7],stride=[2,2], padding = [3,3], bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
# Imporved ResNet18 
        self.in_channels = 64
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = [3,3],stride=[1,1], padding = [1,1], bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
```

具体的ResNet18实现间文件 `resnet_model.py`

### 训练与验证

运行 `run_resnet.py`文件，可以开始训练。

