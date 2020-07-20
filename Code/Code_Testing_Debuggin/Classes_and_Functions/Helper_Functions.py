
import torch

import math

import os

import matplotlib

import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 14, 'text.usetex': True})


def get_num_correct(preds,labels):
       return preds.argmax(dim = 1).eq(labels).sum().item()




@torch.no_grad()
def get_all_preds(model, loader):
       
       '''
       Description:
              - The function get_all_preds will compute 
              all predictions of the data set
       '''
       
       '''
       - the decorator torch.no_grad() will turn off the 
       grad checking of the graph, because when building the
       confusion matrix, we are not doing any training so we don't 
       need PyTorch feature for gradient cheking, so we turn it off
       This will let us save memory
       
       - the model is the network instance
       
       we pass a loder instance beucause when plotting the 
       
       confusion matrix, we can't process all the images at once
       
       so we break them into batches
       '''
       
       all_preds = torch.tensor([])
       for batch in loader:
              images, labels = batch
              
              preds = model(images)
              
              all_preds = torch.cat((all_preds, preds),dim=0)
              
       return all_preds


# For CNN layers
def conv_block_encoder(in_channels, out_channels, parameters_dic):
       
    if parameters_dic['pooling_option'] == True:

           return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 
                        kernel_size = parameters_dic['kernel_size'],
                        padding = parameters_dic['padding']),
        
        
        torch.nn.BatchNorm2d(out_channels),
        
         # activation function choice
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size = parameters_dic['kernel_size_pool'], 
                           stride = parameters_dic['stride_pool'], 
                           padding = parameters_dic['padding_pool']),
   
    )
             
    
    elif parameters_dic['pooling_option'] == False:
            
           return torch.nn.Sequential(
               
        torch.nn.Conv2d(in_channels, out_channels, 
                        kernel_size = parameters_dic['kernel_size'],
                        padding = parameters_dic['padding']),
        
        torch.nn.BatchNorm2d(out_channels),

        # activation function choice
        torch.nn.ReLU(),
   
    )  
       
  
def conv_block_decoder(in_channels, out_channels, parameters_dic):
    
    
        '''
         Here we create the CNN layers for the Decoder Part
        '''
        
        return torch.nn.Sequential(
               
        torch.nn.Conv2d(in_channels, out_channels, 
                        kernel_size = parameters_dic['kernel_size'],
                        padding = parameters_dic['padding']),
        
        torch.nn.BatchNorm2d(out_channels),

        # activation function choice
        torch.nn.ReLU(),
        
        torch.nn.Upsample(scale_factor = parameters_dic['scale_factor'], 
                         mode = parameters_dic['mode_upsampling'])
        
        
        )


    
    
    # For Dense layers
def dense_block(in_features, out_features, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, out_features, **kwargs),
        #nn.BatchNorm2d(out_f),
        
        # activation function choice
        torch.nn.ReLU(),
        
       
    )
       
       
       
def log_CNN_layers(CNN_model):
       
       '''
       This function will print the names of the layers
       along with their shape:
              - how manny filters in each layer
              - depth of each filter
              
       All in batch format
       '''
       
       print('Printing the CNN Model layers name and their corresponding \
             shape: \n')
       
       for name,param in CNN_model.named_parameters():
              
              print(name,'\t \t',param.shape,'\n')
       


def compute_size(parameters_dictionary):
     
           
            '''
             - This method compute the size (width and height) of each 
             feature map outputted from each CNN Hidden layer
           
              
            - We return the height and width of the last CNN layer
            so we can use them when we do a flattening operation
            so we can pass to dense layers
            '''
           
           
            '''
            Unpacking the heigt and width of the input
            '''
            height,width = parameters_dictionary['size_input']
            
            print('--> Initial sizes are: \n height = ',
                  height ,'and width = ', width ,'\n')
            
            
            
            '''
              Looping throught the nb of CNN layers
            '''
            for i in \
           range(len(parameters_dictionary['list_filter_nb_per_layer'])):
                  
                  out_hidden_w = \
                  math.floor((width + 
                     2 * parameters_dictionary['padding'] - 
                     parameters_dictionary['kernel_size'])/
                                parameters_dictionary['stride'] + 1)
                  
                  
                  out_hidden_h = \
                  math.floor((height + 
                     2 * parameters_dictionary['padding'] - 
                     parameters_dictionary['kernel_size'])/
                                parameters_dictionary['stride'] + 1)
                  
                  width,height = out_hidden_w,out_hidden_h
                  
                  
                  print('-> After Conv layer #',i,'\n height = ',
                  height ,'and width = ', width ,'\n')
                  
                  
                  if parameters_dictionary['pooling_option'] == True:
                         
                         
                         out_hidden_w = math.floor((width +
                     2 * parameters_dictionary['padding_pool'] - 
                     parameters_dictionary['kernel_size_pool'])/
                     parameters_dictionary['stride_pool'][1] + 1)
                  
                  
                         out_hidden_h = math.floor((height + 
                     2 * parameters_dictionary['padding_pool'] - 
                     parameters_dictionary['kernel_size_pool'])/
                     parameters_dictionary['stride_pool'][0] + 1)
                         
                         width,height = out_hidden_w,out_hidden_h
         
                         print('-> After pooling layer #',i,'\n height = ',
                               height ,'and width = ', width ,'\n')
                         
                         
                         
def complexity(Neual_Net_model):
    

            nb_parameters = 0 
        
            for name,param in Neual_Net_model.named_parameters():
                
                
              if param.requires_grad == True:
                  
                  print('--> param name is:',name,'\n')
                  
                  print('--> param shape is:',param.shape,'\n')
                
                
              # print('--> param.shape[0]:',param.shape[0],'\n')
              
              
              # print('--> param.shape[1]:',param.shape[1],'\n')
              
              # print('--> param.shape[2]:',param.shape[2],'\n')
              
              # print('--> param.shape[3]:',param.shape[3],'\n')
              
              # nb_parameters = nb_parameters  \
                  
              # + (param.shape[1] * param.shape[2] * param.shape[3] + 1) * param.shape[0]
              
              
              
            # return nb_parameters
              
              
                         
def count_sensors(saving_location_dict):
    
    '''
    This function count the sensor we have in a certain directory
    
    Output: dictionary where:
        
        key: sensor index (string) | value: counts (integer)
        
        Example: '50': 4
    
    '''
    
    
    list_all_numpy_files = os.listdir(saving_location_dict['Directory'])
     
    '''
    Filtering the names to get each data separately
    '''
    list_sensor_name  = [item for item in list_all_numpy_files 
    if item.startswith(saving_location_dict ['File_Name_sensor_id'])]
    
    
    sliced = len('train_id_xx')  
    
    # removing the time information month_day_slice_Nb
    # and taking 'xx' directly by slcing the last 2 elements using
    # [-2:]
    for counter,names in enumerate(list_sensor_name) :
        
        list_sensor_name[counter] = \
        names.replace(names,names[:sliced][-2:])
    
    
    '''
    Creating an empty dictionary for counting
    '''
    
    count_sensor = {}
    
    '''
    Creating the counting
    '''
    for names in list_sensor_name:
        
        if names in count_sensor:
            
            count_sensor[names] += 1
            
        else:
            
            count_sensor[names] = 1
            

    
    return count_sensor
    
    

def plot_results(param_plot):
    
    
    checkpoint_model = torch.load(f"Saved_Iteration/{param_plot['name_file']}")

    frame_result = checkpoint_model['pandas']
    
    
    fig,axes = plt.subplots()
    
    axes.plot(frame_result['Iteration'], frame_result['loss'])
        
    axes.set_title(param_plot['title'])
    axes.set_xlabel(param_plot['xlabel'])
    axes.set_ylabel(param_plot['ylabel'])
        
    axes.grid(color ='b', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
     
    fig.subplots_adjust(left = 0.15, right=.9, bottom = 0.2, top = 0.9)   
    
    fig.savefig(f"Saved_Iteration/{param_plot['save']}")
    
    



def mapper(sensor_dist):
    
    '''
    for each sensor index, we give some label starting from 0
    '''
    
    # Empty dictionary
    mapper = {}

    
    for count,key in enumerate(sensor_dist.keys()) :
    

        mapper[key] = count 
        
        
    return mapper 
                         
                 


                

        
        
     