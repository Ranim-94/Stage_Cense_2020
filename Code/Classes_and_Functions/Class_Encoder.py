

import torch

from Classes_and_Functions.Helper_Functions import conv_block_encoder 



class Encoder(torch.nn.Module):
    
    
    def __init__(self, parameters_neural_network):
           
        super().__init__()
      
        self.parameters_neural_network = parameters_neural_network
        
        
        self._CNN_layers_size = \
        [parameters_neural_network['input_volume'],
         *parameters_neural_network['list_filter_nb_per_layer']]
        
        '''
        Creating the CNN hidden layers:
               in each layer of the CNN, self._CNN_layers_size
               will hold the number of filters in each hidden layer
               of the CNN part
        '''
        
  
        conv_blokcs_list = []
        
        '''
        This loop will create the encoder architecture
        '''
        
        for index, (in_f, out_f) in \
        enumerate(zip(self._CNN_layers_size,self._CNN_layers_size[1:])):
            
            
            if index == len(parameters_neural_network['list_filter_nb_per_layer']) - 1:
                
                '''
                - We have reached the last conv layer
                
                  In this layer, we do global max pooling instead of 
                standard max pooling
                    - so we set the pooling option to be False and
                    create a CNN layer without pooling
                    
                    - The global max pooling will be done in the
                    
                    forward() method using torch.nn.functional.max_pool2d() 
                
                '''

                parameters_neural_network['pooling_option'] = False

                conv_blokcs_list.append(conv_block_encoder(in_f, out_f, 
                                                           parameters_neural_network))
                
                
            else:
                
                '''
                In this block we are in the encoder part:
                    - we create conv layers with standard max pooling layers also
                '''
                
                conv_blokcs_list.append(conv_block_encoder(in_f, out_f, 
                                                           parameters_neural_network))
                

        '''
        --> Creating the Encoder part:      
               - since Sequential does not accept a list, 
                  we decompose the conv_blokcs by using the * operator.
        '''      
        self.enc_block = torch.nn.Sequential(*conv_blokcs_list)
        
        
        
    
    
    