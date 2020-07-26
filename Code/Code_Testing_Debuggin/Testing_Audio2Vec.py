


'''
Experimenting on Audio2Vec 
'''

import torch



from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Helper_Functions import \
log_CNN_layers,compute_size,complexity

from Classes_and_Functions.Class_Custome_Pytorch_Dataset import Dataset_SpecSense

from Classes_and_Functions.Class_Architecture import Model_Architecture

from Classes_and_Functions.Class_Other_Parameters import Other_Parameters

from Classes_and_Functions.Class_Sensor_Classification import My_Calssifier_Encoder

import Classes_and_Functions.Helper_Functions as hf



saving_location_dict = {
    
    'Directory': 'CreatedDataset/Training_Set_25',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



    
'''
--> Creating a dictionary containing different paremeters
for Audio2Vec model
'''
  
# Instantiate the architectures object
model = Model_Architecture()
 
# Accessing the attributes architecture for Audio2Vec 
parameters_dic = model.parameters_Audio2Vec

parameters_dic['batch_size'] = 10



# Dictionary to choose what to test
test_choice = {
    
    # testing custome pythorch dataset for 1 sample
    'sample': False,
    
    
    # testing dataloader for a certain batch
    'batch':False,
    
    'Forward_propagation':False,
    
    'layer_shape':False,
    
    'transfer_learning': True,
    
    # name of the file to be loaded
    'saved_results':'Audio2Vec_emb_350_6.pth',
    
    # Scanning all batches 
    
    'scan_batches':False
    

    }






# ******************** Start Testing **************************************


'''
Loading Data: Instantiate
''' 

dataset_instance = Dataset_SpecSense(saving_location_dict,frame_width = 40,
                                     rows_npy = 10**4,mode = 'pretext_task')


# this will give us an iterable object
train_loader = torch.utils.data.DataLoader(dataset = dataset_instance, 
batch_size = parameters_dic['batch_size'], shuffle = False)


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

    print('********************************************* \n')


# print('----------- Convolution Size Variation through layers \
#       -------------- \n')

# compute_size(parameters_dic)

# print ('---------------------------- \n')


if test_choice['Forward_propagation'] == True:

    # Creating the Neural Network instance
    net_1 = Audio2Vec(parameters_dic)
    
    # convert to an iterator and look at one random sample
    sample,label = next(iter(train_loader))
    
    print('--> Testing forward propagation through encoder part \n')
    
    # Testing Forward Pass
    pred = net_1(sample)
    
    print(f'\t * Prediction shape is {pred.shape} | label shape: {label.shape} \n')

    
    print('********************************************* \n')




# nb_parameters = complexity(net_1)



# pytorch_total_params = sum(p.numel() for p in net_1.parameters() 
#                            if p.requires_grad)


# print('--> nb of parameters is:',pytorch_total_params,'\n')



if test_choice['layer_shape'] == True:
    
    # Creating the Neural Network instance
    net_1 = Audio2Vec(parameters_dic)
    

    print('----------- Audio2Vec Architecture -------------- \n')
    
    log_CNN_layers(net_1)
    
    
    print('********************************************* \n')
    
    
if test_choice['scan_batches'] == True:
    
    print(f'--> Number of balanced Npy files: {len(dataset_instance.spectr_balanced )} ',
          f'| Nb of batches is: {len(train_loader)} \n')
    
    # for count,batch in enumerate(train_loader):
        
    #     print(f'--> batch # {count} \n') 
        
    #     # unpacking
    #     sample , labels = batch 
                                 
    #     print(f' --> Sample shape: {sample.shape} | labels shape: {labels.shape} \n')
                                    
    #     # print(f'--> Labels are: {labels} \n')                         


if test_choice['transfer_learning'] == True:
    
    '''
    When we load the model state dictionary which contains
    the learnable weights and biases, we need to 
    
    redefine the model again then load these learnable 
    weights to the model
    '''
    
    # Creating the Neural Network instance
    net_1 = Audio2Vec(parameters_dic)   
    
    # Loading results
    checkpoint_model = torch.load(test_choice['saved_results'])
    
    # Loading trained weights into the model
    net_1.load_state_dict(checkpoint_model['model_state']) 
    
    
    # Setting the model to training mode
    net_1.train()
    
    # Specifyin the batch size for sensor classification 
    model.parameters_sensor_classification['batch_size'] = 10
    
    # Creating the Neural Network instance for Classification  task
    classif_model = My_Calssifier_Encoder(model.parameters_sensor_classification)
    
    
    # Transfer learning: initilizaing the encoder of the classifier
    # to the same weight values of the trained by Audi2Vec
    classif_model.encoder = net_1.encoder
    
    print('----------- Classification Architecture -------------- \n')
    
    hf.log_CNN_layers(classif_model)
    
    
    print('********************************************* \n')
