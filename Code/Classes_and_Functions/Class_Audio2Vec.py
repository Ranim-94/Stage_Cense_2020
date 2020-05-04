
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
        [parameters_dic['volume_input'],
         *parameters_dic['list_filter_nb_per_layer']]
        
  
        conv_blokcs_list = []
        
        
        for index, (in_f, out_f) in \
        enumerate(zip(self._CNN_layers_size,self._CNN_layers_size[1:])):
            
             
            
            if index == len(parameters_dic['list_filter_nb_per_layer']) - 1:
                
                '''
                - We have reached the last conv layer
                
                
                  In this layer, we do global max pooling instead of 
                standard max pooling
                
                    - global and standard max pooling are the same
                    
                    - the only difference is that in global max pooling,
                    the size of the pooling filter will have the same size
                    as the input feature map
                    
                    - IN other words, the pooling filter height and width
                    are the same as the output feature map 
                    from the last conv layer
                    
                    - By this manner, we will obtain a single number
                    and no need for reshaping operation since global max
                    pooling applied to all the channles in the last conv
                    layer will produce a vector
                    

                '''
                
                
                
                '''
                Changing the size of the pooling filter to make
                it the same as the size of the last feature map
                coming from the last CNN layer
                
                '''
                 
                parameters_dic['pooling_option'] = False
                
        
                conv_blokcs_list.append(conv_block(in_f, out_f, parameters_dic))
                
                
            else:
                
                conv_blokcs_list.append(conv_block(in_f, out_f, parameters_dic))
                


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
        
        self.parameters_dictionary['pooling_option'] = True
        

        out_hidden_w,out_hidden_h = self.__compute_size()
        
        print('-- Done Computing ! \n')
        
        self.parameters_dictionary['kernel_size_pool'] = \
                (out_hidden_h,out_hidden_w * parameters_dic['stride_pool'])
        
        
        # Step 1: Specifying the number of units in each dense layer
        
        
        '''
        Since we have used global max pooling in the last CNN layer,
        
        we don't need to do reshape anymore
       
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

        # # output layer
        # self.last = torch.nn.Linear(self._dense_layers_size[-1], 
        #                       parameters_dic['n_classes'])             
       
    
        self.encoder = torch.nn.Sequential(self.CNN_part,self.dense_part)        
       
       
        
    def forward(self, x):
 
        # Forward Propagation in all CNN layers   
        x = self.CNN_part(x)
        
        print('--> Tensor shape after convolution:',x.shape,'\n')

        '''
        - Reaching the linear part:

        '''
        
        print('--> Kernel Size for global max pooling:',
              self.parameters_dictionary['kernel_size_pool'],'\n')
        
        x = torch.nn.functional.max_pool2d(x,kernel_size = 
            self.parameters_dictionary['kernel_size_pool'])
        
        print('--> Tensor shape after global max pooling:',x.shape,'\n')
        
        x = x.reshape(-1, self._dense_layers_size[0])

        # Proessing in the dense layers part
        x = self.dense_part(x)
        
        
    
        
        
        return x
 
       
    def __compute_size(self):
           
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



          
  
    
       
            
 
