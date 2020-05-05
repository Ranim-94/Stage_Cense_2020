
import torch

import math

from Classes_and_Functions.Helper_Functions import conv_block,dense_block


'''
This class implement the Audio2Vec architecture
'''

class Audio2Vec(torch.nn.Module):
       
    def __init__(self, parameters_dic):
           
        super().__init__()
      
        self.parameters_dictionary = parameters_dic
        
        
        self._CNN_layers_size = \
        [parameters_dic['volume_input'],
         *parameters_dic['list_filter_nb_per_layer']]
        
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
            
            
            if index == len(parameters_dic['list_filter_nb_per_layer']) - 1:
                
                '''
                - We have reached the last conv layer
                
                  In this layer, we do global max pooling instead of 
                standard max pooling
                    - so we set the pooling option to be False and
                    create a CNN layer without pooling
                    
                    - The global max pooling will be done in the
                    
                    forward() method using torch.nn.functional.max_pool2d() 
                
                '''

                parameters_dic['pooling_option'] = False

                conv_blokcs_list.append(conv_block(in_f, out_f, parameters_dic))
                
                
            else:
                
                '''
                In this block we are in the encoder part:
                    - we create conv layers with standard max pooling layers also
                '''
                
                conv_blokcs_list.append(conv_block(in_f, out_f, parameters_dic))
                

        '''
        --> Creating the CNN part:      
               - since Sequential does not accept a list, 
                  we decompose the conv_blokcs by using the * operator.
        '''      
        self.encoder = torch.nn.Sequential(*conv_blokcs_list)
       
 #---------------------- Linear Part ----------------------------------#

        '''
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

       --> So the first step is to set the dimension of the pooling filter
         to be equal to the dimensions of the feature map produced by the 
         last CNN layer
        '''
        
        self.parameters_dictionary['pooling_option'] = True
        

        out_hidden_w,out_hidden_h = self.__compute_size()
        

        '''
        Setting the dimension of the pooling filter
        '''
        self.parameters_dictionary['kernel_size_pool'] = \
        (out_hidden_h,out_hidden_w * parameters_dic['stride_pool'])
                
        '''
        This dimension will be used in the forward method
        when calling torch.nn.functional.max_pool2d()
        '''
        
        
        # Step 2: Specifying the number of units in each dense layer
        
        
        '''
        Since we have used global max pooling in the last CNN layer,
        
        the number of inputs in the last CNN layer will be the same
        of the number of filters used since global max pooling produce 
        
        1 scalar number
       
        '''
        
        self._dense_layers_size = \
        [parameters_dic['list_filter_nb_per_layer'][-1],
         *parameters_dic['dense_layers_list']]
        
        '''
        self._dense_layers_size[0]: contains the number
        of flattened features
        '''
        
        
        dense_blokcs_list = [dense_block(in_f, out_f) 
                       for in_f, out_f in zip(self._dense_layers_size, 
                                              self._dense_layers_size[1:])]
                       
        self.dense_part = torch.nn.Sequential(*dense_blokcs_list)

       
        
    def forward(self, x):
 
        # Forward Propagation in all CNN layers   
        x = self.encoder(x)
        

        '''
        - Reaching the linear part:
            - need to do global max pooling first
        '''
     
        x = torch.nn.functional.max_pool2d(x,kernel_size = 
            self.parameters_dictionary['kernel_size_pool'])
        

        '''
        Transform the batch format into rank 2 tensor
        '''
        x = x.reshape(-1, self._dense_layers_size[0])

        # Proessing in the dense layers part
        x = self.dense_part(x)
        
    
        return x
 
       
    def __compute_size(self):
           
            '''
             - This method compute the size (width and height) of each 
             feature map outputted from each CNN Hidden layer
                 - If there is pooling layer after convolution,
                 also the dimension will be computed
           
              
            - We return the height and width of the last CNN layer
            so we can use them when we do a flattening operation
            so we can pass to dense layers
            '''
           
           
            '''
            Unpacking the heigt and width of the input
            '''
            width, height = self.parameters_dictionary['size_input']
    
           
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
                         
                       

            '''
              This will be the size of the feature map
              coming out from the last CNN layer we have in our
              model
            '''       
            return out_hidden_w,out_hidden_h



          
  
    
       
            
 
