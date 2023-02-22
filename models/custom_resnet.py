import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
      self.prep_layer = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
      ) # 32 | 32

      # Layer1 - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k], R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k], Add(X, R1)
      self.layer1_x  = self.layer_X (64, 128) # 32 | 32 | 16
      self.layer1_res =  self.layer_resnet(128, 128) # 16 | 16

      # Layer 2 - Layer 2 - Conv 3x3 [256k] MaxPooling2D BN ReLU
      self.layer2 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding = 1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(256),
        nn.ReLU()
      ) # 16 | 16 | 8

      # Layer 3 - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k] R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k] Add(X, R2)
      self.layer3_x = self.layer_X(256, 512) # 8 | 8 | 4
      self.layer3_res = self.layer_resnet(512, 512) # 4 | 4

      # MaxPooling with Kernel Size 4
      self.pool = nn.MaxPool2d(4, 4) # 4 | 1

      # FC Layer
      self.fc = nn.Linear(512, 10,bias=False)


    def layer_X(self, in_channels, out_channels):
      layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
      )
      return layer

    def layer_resnet(self, in_channels, out_channels):
      res_layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
      )
      return res_layer

    def forward(self, x):
      x = self.prep_layer(x)

      x = self.layer1_x(x)
      res1  = self.layer1_res(x)
      x = x + res1

      x = self.layer2(x)

      x = self.layer3_x(x)
      res2 = self.layer3_res(x)
      x = x + res2

      x = self.pool(x)
      #print ("here1:",x.shape)

      x = x.view(x.size(0), -1)
      #print ("here2:", x.shape)
      x = self.fc(x)
      return F.softmax(x, dim=-1)#SoftMax
