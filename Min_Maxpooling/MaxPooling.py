'''
Le pooling, lui, sert principalement à réduire la taille spatiale des feature maps
'''

# import des librairies
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


#create a pool class instance with params
poolsize=3
stride=3
p1=nn.MaxPool2d(poolsize,stride)

p2=nn.MaxPool3d(poolsize,stride)

# let's have a look at them
print(p1)
print(p2)


#create a image and apply MaxPooling
#image 1D and 2D
img1=torch.rand(1,1,30,30)
img2=torch.rand(1,3,30,30)


# all combinations of image and maxpool dimensionality

image1pool=p1(img1)
print(image1pool.shape)

image2pool=p1(img2)
print(image2pool.shape)

image2pool1=p2(img2)
print(image2pool1.shape)

#create a simple CNN

littlenet= nn.Sequential(
  
  nn.Conv2d(3,10,5,3,2),# convolution layer
  nn.ReLU(),#activation function
  nn.AvgPool3d(3,3), #average Pool
  
  
  # the FFN block
    nn.Flatten(),          # vectorize to get from image to linear
    nn.Linear(588,1),      # FC linear layer
    nn.Sigmoid()           # output activation

)

#test with a bit of data 

image=torch.rand(1,3,128,128)
littlenet(image)
