import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # Convolution 1 e.g. 32x32x3 | 3x3x3x16  
      self.conv1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU()
      ) 

      # Convolution 2 | 3x3x16x32
      self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU()
      )

      # Convolution 3 | 3x3x32x48
      self.conv3 = nn.Sequential(
        nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU()
      ) 

      # Apply GAP and get 1x1x48
      self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=32)
      ) # output_size = 1
      
      # FC Layer ultimus 
      self.ultimus_layer = nn.Linear(48, 8,bias=False)
      self.ultimus_rev = nn.Linear(8, 48,bias=False)
     
      # FC layer that converts 48 to 10 
      self.out = nn.Linear(48, 10, bias = True)

    # Ultimus block 
    # Creates 3 FC layers called K, Q and V such that: X*K = 48*48x8 > 8,  X*Q = 48*48x8 > 8, X*V = 48*48x8 > 8 
    # then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
    # then Z = V*AM = 8*8 > 8
    # then another FC layer called Out that:
    # Z*Out = 8*8x48 > 48
    def Ultimus (self, input):
      k = self.ultimus_layer(input)
      q = self.ultimus_layer(input)
      v = self.ultimus_layer(input)
      #print (k)
      #create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
      AM =  F.softmax(torch.div(torch.matmul(torch.transpose(q, 0, 1), k) , pow(8, 0.5)))
      
      # then Z = V*AM = 8*8 > 8
      Z = torch.matmul(v, AM)
      z_out = self.ultimus_rev(Z)
      return z_out
      
     
    def forward(self, x):
      # 3 convolution layers
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)

      # Gap layer 
      x = self.gap(x)
      #print(x.shape)
      x = x.view(x.size(0), -1)
      
      # Repeat Ultimus block 4 times
      x = self.Ultimus(x)  
      x = self.Ultimus(x)  
      x = self.Ultimus(x)  
      x = self.Ultimus(x)      

      #FC layer that converts 48 to 10 
      x = self.out(x)
      return F.softmax(x, dim=-1)#SoftMax
