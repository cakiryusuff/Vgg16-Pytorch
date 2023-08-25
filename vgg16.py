# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:37:24 2023

@author: cakir
"""

import torch.nn as nn

class Vgg16(nn.Module):
    def __init__(self, num_classes = 1000):
      super(Vgg16, self).__init__()
      self.planes = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

      self.network = self._make_layer(self.planes)
      self.avgPool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
      self.fc = nn.Sequential(
          nn.Linear(512*7*7, 4096),
          nn.ReLU(),
          nn.Dropout(),

          nn.Linear(4096, 4096),
          nn.ReLU(),
          nn.Dropout(),

          nn.Linear(4096, num_classes))

    def _make_layer(self, planes):
      layers = []
      for i in range(len(self.planes) - 1):
          layers.append(self.layer(planes[i], planes[i + 1]))
          if i in [1, 3, 6, 9, 12]:
            layers.append(nn.MaxPool2d(2, 2))
      return (nn.Sequential(*layers))

    def layer(self, in_planes, out_planes):
      blocks = nn.Sequential(
          nn.Conv2d(in_planes, out_planes, kernel_size = 3, padding = 1, bias = False),
          nn.ReLU(inplace = True),
          nn.BatchNorm2d(out_planes))
      return blocks

    def forward(self, x):
      x = self.avgPool(self.network(x))
      x = x.view(x.size()[0], -1)
      x = self.fc(x)
      x = nn.functional.softmax(x, dim = 1)
      return x
