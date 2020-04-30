
import torch

import torch.nn.functional as F


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
def conv_block(in_channels, out_channels, parameters_dic):
       
    if parameters_dic['pooling_option'] == True:

           return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 
                        kernel_size = parameters_dic['kernel_size']),
        torch.nn.BatchNorm2d(out_channels),
        
         # activation function choice
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size = parameters_dic['kernel_size_pool'], 
                           stride = parameters_dic['stride_pool'], 
                           padding = parameters_dic['padding_pool']),
        
       
   
    )
    
    else:
           return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 
                        kernel_size = parameters_dic['kernel_size']),
        torch.nn.BatchNorm2d(out_channels),

        # activation function choice
        torch.nn.ReLU(),
   
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
       


        

        
        
     