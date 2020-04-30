


'''
Experimenting on Audio2Vec 
'''

import torch

import random

import torchvision

import torchvision.transforms as transforms
from torchvision import transforms, datasets

from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Helper_Functions import log_CNN_layers




# Loading the Dataset

train_set = torchvision.datasets.FashionMNIST( 
              root='./data/FashionMNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

'''
--> Creating a dictionary containing different paremeters
'''

parameters_dic = {
       
       # Input layer
       
       # [width, height]
       'size_input': [28,4],
       
       # If we are working with gray scale of RGB images
       'volume_input':1,
       #---------------------------------------
       
       # for CNN layers
       'list_filter_nb_per_layer':[64,128,256,256],
       
       'padding':0, 'stride': 1,'kernel_size':3,
       
       #---------------------------------------
       
       # for pooling layers
       
       'pooling_option':True,
       
       'padding_pool':0, 'stride_pool': 2,'kernel_size_pool':2,
       #---------------------------------------
       
       # for Multilayer perceptrons part
       'dense_layers_list' : [128],

       }



# Accessing an image sample from training set

sample = next(iter(train_set))

# unpacking

image , _ = sample

start , finish = 3 , image.shape[1] - 2


num1 = random.randint(start , finish)

print("- Random integer: ", num1,'\n')


print('- Dummy sample before slicing is :',image.shape,'\n')


# slicing
sample = image[:,num1 - 2 : num1 + 2,:].clone()


print('- Dummy sample after slicing is :',sample.shape,'\n')


print('- sample shape after squeezing is :',
      sample.unsqueeze(0).shape,'\n')




print('----------- Convolution Size Variation through layers \
      -------------- \n')

# Creating the Neural Network instance
net_1 = Audio2Vec(parameters_dic)

print('----------- Encoder Architecture -------------- \n')

log_CNN_layers(net_1)


print ('---------------------------- \n')


print('-- > Testing forward propagation through encoder part \n')

# Testing Forward Pass
pred = net_1(sample.unsqueeze(0))

# print('- Prediction shape is',pred.shape,'\n')

