import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.nn.functional as func
import random


class DigitRecognizer(nn.Module):
  def __init__(self):
    super(DigitRecognizer, self).__init__()
    self.conv1 = nn.Conv2d(1, 5, 5)
    self.pool = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(5, 10, 5)
    self.linear1 = nn.Linear(10*4*4, 10)

  def forward(self, x):
    x = self.pool(func.relu(self.conv1(x)))
    x = self.pool(func.relu(self.conv2(x)))
    x = x.view(-1, 10*4*4)
    x = func.softmax(self.linear1(x), dim=1)
    return x

# model = DigitRecognizer()
# model.load_state_dict(torch.load("MNIST.pth"))
# model.eval()
#
# data_train = list(datasets.MNIST('data',train=True, download=True,transform=transforms.ToTensor()))[:40000]
# sample = data_train[0][0]
