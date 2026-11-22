'''
    This file contains the primary model architecture, which will consist of a CNN,
    along with an RNN after the image is processed, and a few fully-connected layers
    to finish the network. It willa also contain the training loop and stats code for
    model training, and validation in terms of the outputs. stats_calc.py will contain
    distance-related calculations.
'''

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np

class FetchNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional portion: Go from 3 to 64 channels, from a 96x96 image down to 
        # 8x8 to capture as many features as possible
        self.conv1 == nn.Conv2d(3, 8, kernel_size=3) # 96x96 --> 94x94
        self.pool = nn.MaxPool2d(2, 2) # 94x94 --> 47x47
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3) # 47x47 --> 45x45 --> 22x22
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3) # 22x22 --> 20x20 --> 10x10
        self.conv4 = nn.Conv2d(32, 64, kernel_size = 3) # 10x10 --> 8x8

        # Pass these through a fully-connected layer before giving it to the RNN
        self.conv_fc = nn.Linear(64*8*8, 64)
