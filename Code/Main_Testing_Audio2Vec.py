


'''
Experimenting on Audio2Vec 
'''

import torch

import numpy as np

import random

import torchvision

import torchvision.transforms as transforms
from torchvision import transforms, datasets

from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Helper_Functions import \
log_CNN_layers,compute_size




# Loading Some Dummy Variable

sample_original = \
torch.from_numpy(np.load('train_spec_0_12_6_0.npy')).float()

'''
This casting is because numpy use float64 as their default type

We need to cast the tensor to double so the data and the model
have same data type
'''


print('- Shape of the sample_original is:',sample_original.shape,'\n')

'''
--> Creating a dictionary containing different paremeters
for Audio2Vec model
'''

parameters_dic = {
       
       # Input layer
       
       # [width, height]
       'size_input': (29,1),
       
       # If we are working with gray scale of RGB images
       'volume_input':1,
       #---------------------------------------
       
       # for CNN layers
       'list_filter_nb_per_layer':(64,128,256,256),
       
       'padding':1, 'stride': 1,'kernel_size':3,
       
       #---------------------------------------
       
       # for pooling layers
       
       'pooling_option':True,
       
       'padding_pool':0, 'stride_pool': 2,'kernel_size_pool':1,
       #---------------------------------------
       
       # for Multilayer perceptrons part: number of neurons
       # for each dense fully connected layer
       'dense_layers_list' : (128,),

       }


start , finish =  3 , sample_original.shape[1] - 2

num1 = random.randint(start,finish )

print("- Random integer: ", num1,'\n')

list_trial = [*parameters_dic['list_filter_nb_per_layer']]




'''
Slice the tensor and truning into a batch format 
so we can process it into the Audio2Vec model

    - batch format shape = [batch size , volume , Height, width]
'''
# sample = sample_original[num1 - 2 : 
#                          num1 + 2,:].clone().unsqueeze(0).unsqueeze(0)


sample = sample_original[num1 - 1,:].reshape(-1,29).clone().\
unsqueeze(0).unsqueeze(0)



print('- Sample shape after squeezing and slicing is :',
      sample.shape,'\n')


print('----------- Convolution Size Variation through layers \
      -------------- \n')

compute_size(parameters_dic)

print ('---------------------------- \n')



# Creating the Neural Network instance
net_1 = Audio2Vec(parameters_dic)



print('----------- Encoder Architecture -------------- \n')

log_CNN_layers(net_1)


print ('---------------------------- \n')




print('-- > Testing forward propagation through encoder part \n')

# Testing Forward Pass
pred = net_1(sample)

print('- Prediction shape is',pred.shape,'\n')

