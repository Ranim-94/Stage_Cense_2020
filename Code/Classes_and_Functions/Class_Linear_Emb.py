

import torch

from Classes_and_Functions.Helper_Functions import dense_block

class Linear_Emb(torch.nn.Module):
    
    def __init__(self, parameters_neural_network):
        
        
        super().__init__()
        
        '''      
        Some Explanation:
        -------------------
        
        --> Usually when reached the linear part we         
        do a flatten opeartion by computing the number of features
        map produced in the last CNN layer and turing them into 1 D vector
        
        
            - number of features in the last CNN layer  =
            
                number of filters used * heigth_feature_map * 
                    width_feature_map
                    
        However in Audio2Vec the authors used global max pooling
        
        
         - global and standard max pooling are the same
                    
         - the only difference is that in global max pooling,
         the size of the pooling filter will have the same size
         as the input feature map
         
         - And global max pooling produce a scalar number (not an image map)

        
        --> Since the authors used global max pooling in the last CNN layer,
            there is no need to compute the dimension of the feature map
            since global max pooling produce 1 scalar number
      
        '''
        
        self._dense_layers_size = \
        [parameters_neural_network['list_filter_nb_per_layer'][-1],
         *parameters_neural_network['dense_layers_list']]
        
        '''
        self._dense_layers_size[0]: contains the number
        of flattened features
        '''
        
        
        dense_blokcs_list = [dense_block(in_f, out_f) 
                       for in_f, out_f in zip(self._dense_layers_size, 
                                              self._dense_layers_size[1:])]
                       
        self.dense_part = torch.nn.Sequential(*dense_blokcs_list)
        
        
        
        
        
        