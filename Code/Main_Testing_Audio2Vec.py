


'''
Experimenting on Audio2Vec 
'''

import torch

import numpy as np





from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Helper_Functions import \
log_CNN_layers,compute_size,complexity

from Classes_and_Functions.Class_Custome_Pytorch_Dataset import Dataset_SpecSense


'''
--> Creating a dictionary containing different paremeters
for Audio2Vec model
'''

parameters_dic = {
       
       # Input layer
       
       # [height, width]
       'size_input': (4,29),
       
       # If we are working with gray scale of RGB images
       'volume_input':1,
       #---------------------------------------
       
       # for CNN layers
       'list_filter_nb_per_layer':(64,128,256,256,512,512),
       
       'padding':1, 'stride': 1,'kernel_size':3,
       
       #---------------------------------------
       
       # for pooling layers
       
       'pooling_option':True,
       
       'padding_pool':0, 'stride_pool': (1,2),'kernel_size_pool':1,
       
       # stride_pool: (for striding height, for striding width)
       
       #---------------------------------------
       
       # for Multilayer perceptrons part: number of neurons
       # for each dense fully connected layer
       'dense_layers_list' : (128,),
       
       
       # ---------- Decoder Specification -------------
       
       
       'mode_upsampling':'nearest', 'scale_factor':(1,0.79),
       
       # 'scale_factor':(height,width)
       
        'scale_reconstruction':(0.3,1),
        
        
        #------------- Batch Size -------------------
        
        'batch_size':10
       

       }


data_percentage = 25

saving_location_dict = {
    
     'Directory': 'CreatedDataset/Training_Set_50_57',
    
    # 'Directory': '/media/ranim/Seagate Expansion Drive/Training_Set',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



'''
Loading Data: Instantiate
'''

dataset_instance = Dataset_SpecSense(saving_location_dict,data_percentage)


# this will give us an iterable object
train_loader = torch.utils.data.DataLoader(dataset = dataset_instance, 
                          batch_size = parameters_dic['batch_size'],
                          shuffle = True)



# convert to an iterator and look at one random sample
sample,label = next(iter(train_loader))




print('---> Sample shape is :',sample.shape,'\n')


print('---> Label shape is :',label.shape,'\n')




# print('----------- Convolution Size Variation through layers \
#       -------------- \n')

# compute_size(parameters_dic)

# print ('---------------------------- \n')



# Creating the Neural Network instance
net_1 = Audio2Vec(parameters_dic)



print('-- > Testing forward propagation through encoder part \n')

# Testing Forward Pass
pred, embedding = net_1(sample)

print('--> Prediction shape is',pred.shape,'\n')


print('--> Embedding shape is:',embedding.shape,'\n')




# nb_parameters = complexity(net_1)



# pytorch_total_params = sum(p.numel() for p in net_1.parameters() 
#                            if p.requires_grad)


# print('--> nb of parameters is:',pytorch_total_params,'\n')


# print('----------- Audio2Vec Architecture -------------- \n')

# log_CNN_layers(net_1)


# print ('---------------------------- \n')



