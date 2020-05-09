
import torch



from Classes_and_Functions.Helper_Functions import conv_block_encoder,\
dense_block, conv_block_decoder


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

                conv_blokcs_list.append(conv_block_encoder(in_f, out_f, 
                                                           parameters_dic))
                
                
            else:
                
                '''
                In this block we are in the encoder part:
                    - we create conv layers with standard max pooling layers also
                '''
                
                conv_blokcs_list.append(conv_block_encoder(in_f, out_f, 
                                                           parameters_dic))
                

        '''
        --> Creating the Encoder part:      
               - since Sequential does not accept a list, 
                  we decompose the conv_blokcs by using the * operator.
        '''      
        self.encoder = torch.nn.Sequential(*conv_blokcs_list)
       
 #---------------------- Linear and Embedding Part ----------------#

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

 
#------------------------------ Decoder Part ---------------------------

        
        '''
        Setting the input channel to 1 for the decoder part
        '''

        self._CNN_layers_size[-1] = 1
        
        '''
        Reversing the order of the layers for the Decoder Part
        '''
        self._CNN_layers_size.reverse()
        
        
        decoder_blokcs_list = [conv_block_decoder(in_f, out_f,parameters_dic) 
                       for in_f, out_f in zip(self._CNN_layers_size, 
                                              self._CNN_layers_size[1:])]
        
  
        self.decoder = torch.nn.Sequential(*decoder_blokcs_list)
        
        
        self.decoder_restore =  \
        torch.nn.Upsample(scale_factor = 
                          parameters_dic ['scale_reconstruction'], 
                    mode = parameters_dic['mode_upsampling'])
        
        
 


#---------------------- Methods Imeplementation -----------------------
      
        
    def forward(self, x):
  
        
        # Forward Propagation in all CNN layers   
        x = self.encoder(x)
        

        '''
        - Reaching the linear part:
            - need to do global max pooling first
        '''
     
        x = torch.nn.functional.max_pool2d(x,
        kernel_size = (1,x.shape[3]) )
        
        '''
        Since we have input all frames (2 of the past and 2 of the future)
        
        the global max pooling need to be performed on each frame alone,
        
        where each frame have a dimension 1 x something
            - where something is the result coming from the processing
                through the encoder part.
            - that's why we have set the kernel_size to (1,x.shape[3])
            
            and not (x.shape[2],x.shape[3])
        '''
        

        '''
        Transform the batch format into rank 2 tensor
        to be processed in the fully connected layers
        '''
        x = x.reshape(-1, self._dense_layers_size[0])
        

        # Proessing in the dense layers part
        x = self.dense_part(x)
        
        
        '''
        Storing the embedding results in a variable 
        so we can take and use it later in the downstream task
        '''
        embedding = x.clone()
        
        
        # Start Processing theough the Decoder
        
        '''
        First we need to adjust again the format of the tensor
        to a rank 4 tensor
        '''

        
        x = \
        x.reshape(self.parameters_dictionary['batch_size'],
                  1,self.parameters_dictionary['size_input'][0],
                  self.parameters_dictionary['dense_layers_list'][-1])
        
        
        x = self.decoder(x)
        
        x = self.decoder_restore(x)
        

        return x,embedding
 
       
 
  
    
       
            
 
