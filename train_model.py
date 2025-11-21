import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
#should use sklearn.preprocessing....
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # Input channels, output channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convT1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.convT2 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid() # For image data scaled between 0 and 1

    def forward(self, x):
        x = self.relu(self.convT1(x))
        x = self.sigmoid(self.convT2(x))
        return x


#save model, maybe using pickle??
