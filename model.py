from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net1(nn.Module):
    def __init__(self, drop=0.01):
        super(Net1, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
        ) 


        # TRANSITION BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), # output_size = 10    RF:  9
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
            
        ) 

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # output_size = 10    RF:  13
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
            
        )
        self.convblock5 = nn.Sequential(
           nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # output_size = 10    RF:  17
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
            
        )

        
        # Global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(8)  
        )
        
        # Fully connected layer
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x) #3
        x = self.convblock2(x) #5
        x = self.pool(x) #6
        x = self.trans1(x)
        x = self.convblock3(x) # 10
        x = self.convblock4(x) # 14
        x = self.convblock5(x) # 18
        x = self.gap(x)
        x = self.convblock6(x) 
        x = x.view(-1, 10)   # convert 2D to 1D
        
        return F.log_softmax(x, dim=-1)
    