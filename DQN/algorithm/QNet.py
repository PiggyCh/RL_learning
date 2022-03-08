import torch
import torch.nn as nn
class Q_net(nn.Module):
    def __init__(self, Dim_in, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=Dim_in,out_channels=32,kernel_size=(8,8),stride=(4,4)) #(84-8)/4 +1 =76/4 +1 =20
        self.maxpool1 =nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2)) # (20-4)/2 +1 =9
        self.maxpool2 = nn.MaxPool2d(2, stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1)) # (9-3)/1 +1 =7  7*7*64
        self.fc1   = nn.Linear(in_features=256,out_features=256)
        self.fc2   = nn.Linear(in_features=256,out_features=act_dim)
        self.Relu =nn.ReLU()
    def forward(self,x): # inshape (batch,x,y,channel)
        x = self.conv1(x)
        x=  self.Relu(x)
        #
        x = self.Relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.Relu(self.conv3(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0),-1)
        x = self.Relu(self.fc1(x))
        x = self.fc2(x)
        return x