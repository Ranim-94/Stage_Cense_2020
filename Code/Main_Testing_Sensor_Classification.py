






'''
Using Encoder part of Audio2Vec for Sensor Classification 
'''

import torch

import Classes_and_Functions.Helper_Functions as hf

from Classes_and_Functions.Class_Custome_Pytorch_Dataset import Dataset_SpecSense

from Classes_and_Functions.Class_Sensor_Classification import My_Calssifier_Encoder







'''
--> Creating a dictionary containing different paremeters
for Sensor Classification model
'''

parameters_neural_network = {
       
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
       
       # nb of classes we are trying to classify: in our case nb of sensors
       'n_classes' : 7,
       
       #------------- Batch Size -------------------
       
       'batch_size':20

       }



saving_location_dict = {
    
    # 'Directory': 'Training_Set',
    
    'Directory': 'CreatedDataset/Training_Set_50_57',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



# Dictionary to choose what to test
test_choice = {
    
    # testing custome pythorch dataset for 1 sample
    'sample': False,
       
    # testing dataloader for a certain batch
    'batch':False,
    
    'Forward_propagation':False,
    
    'layer_shape':False,
    
    'dataloader':False,
    
    'save_load':True, 
    
    # Model name to be tested
    'path': 'weights_0.pth'
    
    }


data_percentage = 25


#**************************** Start Testing ********************************


'''
Loading Data: Instantiate
'''  

dataset_instance = Dataset_SpecSense(saving_location_dict,data_percentage,
                                     mode = 'sensor_classification')

# this will give us an iterable object
train_loader = torch.utils.data.DataLoader(dataset = dataset_instance, 
               batch_size = parameters_neural_network['batch_size'],
               shuffle = True)
    




if test_choice['sample'] == True:
    
    # tesing getitem()
    sample,  label = dataset_instance[4]
    
    print('********************************************* \n')
    
    print('--> Test 1 sample building \n')

    print('---> Sample shape is :',sample.shape,'\n')
    
    
    print('---> Label shape is :',label.shape,'\n')

    print('********************************************* \n')

if test_choice['batch'] == True:


    # convert to an iterator and look at one random sample
    sample,label = next(iter(train_loader))
    
    
    print('********************************************* \n')
    
    print('--> Testing batch building \n')
    
    
    print('---> Sample shape is :',sample.shape,'\n')
    
    
    print('---> Label shape is :',label.shape,'\n')
    
    print(f'--> Label is : \n {label}')

    print('********************************************* \n')
    
    

if test_choice['Forward_propagation'] == True:

    # Creating the Neural Network instance
    net_1 = My_Calssifier_Encoder(parameters_neural_network)
    
    
    # convert to an iterator and look at one random sample
    sample,label = next(iter(train_loader))
    
    print('-- > Testing forward propagation through encoder part \n')
    
    # Testing Forward Pass
    pred = net_1(sample)
    
    print('--> Prediction shape is',pred.shape,'\n')
    

    print('********************************************* \n')
    
    
if test_choice['layer_shape'] == True:
    
    
    # Creating the Neural Network instance
    net_1 = My_Calssifier_Encoder(parameters_neural_network)

    print('----------- Encoder Classification Architecture -------------- \n')
    
    hf.log_CNN_layers(net_1)
    
    
    print('********************************************* \n')



if test_choice['dataloader'] == True:
    
    print(f'Nb of batches in train loader is: {len(train_loader)} \n')
    
    for count,batch in enumerate(train_loader):
        
        print(f'--> batch # {count} \n') 
        
        
        # unpacking
        sample , labels = batch 
        
        print(f'Label is: {labels} \n')
        
        labels = labels.reshape(-1)
        
        print(f'-->Sample shape is: {sample.shape}',
              f'| Labels shape is: {labels.shape} \n')
        
        
        
if test_choice['save_load'] == True:
    
    '''
    When we load the model state dictionary which contains
    the learnable weights and biases, we need to 
    
    redefine the model again then load these learnable 
    weights to the model
    '''
    
    # Creating the Neural Network instance
    net_1 = My_Calssifier_Encoder(parameters_neural_network)   
    
    net_1.load_state_dict(torch.load(test_choice['path']))
    
    
    # Setting the model to evaluation mode
    net_1.eval() 
    
    print('----------- Encoder Classification Architecture -------------- \n')
    
    hf.log_CNN_layers(net_1)
    
    
    print('********************************************* \n')

        
        








