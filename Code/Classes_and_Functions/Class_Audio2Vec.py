
import torch

import math

from Classes_and_Functions.Helper_Functions import conv_block,dense_block




'''
The purpose of this module is to get rid of the conv_block

and creating them using a loop, so we don't hard code them

Also, the hidden layers size (number of filters in each layers) will also

be inputted and not hard coded
'''

class Audio2Vec(torch.nn.Module):
       
    def __init__(self, parameters_dic):
           
        super().__init__()
      
        self.parameters_dictionary = parameters_dic
        
        '''
        Creating the CNN hidden layers:
               in each layer of the CNN, self._CNN_layers_size
               will hold the number of filters in each hidden layer
               of the CNN part
        '''
        self._CNN_layers_size = \
        [parameters_dic['volume_input'] ,  
        *parameters_dic['list_filter_nb_per_layer'] ]
        
        '''
        - Creating the conv_block using list comprehension and the zip
        function from itertool
        
        - conv_blocks will be a list of CNN layer
        '''
        
        conv_blokcs_list = \
        [conv_block(in_f, out_f, parameters_dic) 
                       for in_f, out_f in zip(self._CNN_layers_size, 
                                              self._CNN_layers_size[1:])]

        '''
        --> Creating the CNN part:
               
               - since Sequential does not accept a list, 
                  we decompose the conv_blokcs by using the * operator.
        '''
       
        self.CNN_part = torch.nn.Sequential(*conv_blokcs_list)
       
 #---------------------- Linear Part ----------------------------------#

        '''
        -->Dense layer part:
               - we will do the same thing we did for the CNN part
        '''
        
        # Step 1: Specifying the number of units in each dense layer
        
        '''
        Don't forget that we need to flatten before we go to the 
        
        dense part
        '''
        
        # Computing the size for flattening part
        
        out_hidden_last_CNN_layer = self.compute_size()
        
        
        '''
        out_hidden_last_CNN_layer is of type tuple
               - it contains the width and height 
               of the feature map produced by the last CNN hidden
               layer
        '''
        
        self._dense_layers_size = \
        [parameters_dic['list_filter_nb_per_layer'][-1] * 
         out_hidden_last_CNN_layer[0] * out_hidden_last_CNN_layer[1] ,
         *parameters_dic['dense_layers_list']]
        
        '''
        self._dense_layers_size[0]: contains the number
        of flattened features
        '''
        
        
        dense_blokcs_list = [dense_block(in_f, out_f) 
                       for in_f, out_f in zip(self._dense_layers_size, 
                                              self._dense_layers_size[1:])]
                       
        self.dense_part = torch.nn.Sequential(*dense_blokcs_list)

        # # output layer
        # self.last = torch.nn.Linear(self._dense_layers_size[-1], 
        #                       parameters_dic['n_classes'])             
       
        
    def forward(self, x):
 
        # Forward Propagation in all CNN layers   
        x = self.CNN_part(x)
        

        '''
        - Reaching the linear part:
               - First we flatten then pass
                to the dense layers
        '''
        # Flattening
        x = x.reshape(-1, self._dense_layers_size[0]) 
        
        # Proessing in the dense layers part
        x = self.dense_part(x)
        
        return x
 
       
    def compute_size(self):
           
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
            width, height = self.parameters_dictionary['size_input']
            
            print('--> Initial sizes are: \n width = ',
                  width, 'and height = ',height,'\n')
            
            
            
            '''
              Looping throught the nb of CNN layers
            '''
            for i in \
           range(len(self.parameters_dictionary['list_filter_nb_per_layer'])):
                  
                  out_hidden_w = \
                  math.floor((width + 
                     2 * self.parameters_dictionary['padding'] - 
                     self.parameters_dictionary['kernel_size'])/
                                self.parameters_dictionary['stride'] + 1)
                  
                  
                  out_hidden_h = \
                  math.floor((height + 
                     2 * self.parameters_dictionary['padding'] - 
                     self.parameters_dictionary['kernel_size'])/
                                self.parameters_dictionary['stride'] + 1)
                  
                  width,height = out_hidden_w,out_hidden_h
                  
                  print('--> After Conv layer #',i,'\n width = ',
                        width,'and height = ',height,'\n')
                  
                  
                  if self.parameters_dictionary['pooling_option'] == True:
                         
                         
                         out_hidden_w = math.floor((width +
                     2 * self.parameters_dictionary['padding_pool'] - 
                     self.parameters_dictionary['kernel_size_pool'])/
                     self.parameters_dictionary['stride_pool'] + 1)
                  
                  
                         out_hidden_h = math.floor((height + 
                     2 * self.parameters_dictionary['padding_pool'] - 
                     self.parameters_dictionary['kernel_size_pool'])/
                     self.parameters_dictionary['stride_pool'] + 1)
                         
                         width,height = out_hidden_w,out_hidden_h
                         
                         print('--> After pooling layer #',i,'\n width = ',
                        width,'and height = ',height,'\n')
                 

            '''
              This will be the size of the feature map
              coming out from the last CNN layer we have in our
              model
            '''       
            return out_hidden_w,out_hidden_h



          
  
    
       
            
 
