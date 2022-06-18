# coding: utf-8
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # 父类构造方法
        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),            # [3,128,128]->[64,128,128]
            nn.BatchNorm2d(64),                   
            nn.ReLU(), 
            nn.MaxPool2d(2, 2, 0),                # [64,128,128]->[64,64,64]

            nn.Conv2d(64, 128, 3, 1, 1),          #[64,64,64]->[128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                #[128,32,32]->[128,32,32]

            nn.Conv2d(128, 256, 3, 1, 1),         #[128,32,32]->[256,32,32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),                #[256,32,32]->[256,8,8]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x




