

import torch



from Classes_and_Functions.Helper_Functions import conv_block_encoder,\
dense_block, conv_block_decoder





from Classes_and_Functions.Class_Linear_Emb import Linear_Emb

'''
This class implement the Audio2Vec architecture
'''

class Audio2Vec(torch.nn.Module):
       
    def __init__(self, parameters_neural_network):
           
        super().__init__()
      
        self.parameters_neural_network = parameters_neural_network
        
        
        self._CNN_layers_size = \
        [parameters_neural_network['input_volume'] ,
         *parameters_neural_network['list_filter_nb_per_layer'] ]
        
        
        print(f'Encoder Part : {self._CNN_layers_size} \n')
        
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
        
        self.encoder = torch.nn.Sequential(*conv_blokcs_list)
           
        
       
 #---------------------- Linear and Embedding Part ----------------#

        # Instanting
        lin_emb = Linear_Emb(parameters_neural_network)
                       
        self.dense_part  = lin_emb.dense_part
        
        
        # Needed in forward() method
        self._dense_layers_size = lin_emb._dense_layers_size

 
# ------------------------------ Decoder Part ---------------------------

        
        
        # Setting the volum of the input channel to 1 for the decoder part
        self._CNN_layers_size.append(1) 
        
        
        
        
        '''
        Reversing the order of the layers for the Decoder Part
        Since the decorder Architecture is the mirror of the 
        Encoder part
        '''
        
        self._CNN_layers_size.reverse()
        
        print(f'{self._CNN_layers_size} \n')
        
        
        decoder_blokcs_list = [conv_block_decoder(in_f, out_f,parameters_neural_network) 
                       for in_f, out_f in zip(self._CNN_layers_size, 
                                              self._CNN_layers_size[1:])]
        
  
        self.decoder = torch.nn.Sequential(*decoder_blokcs_list)
        
        
        self.decoder_restore =  \
        torch.nn.Upsample(scale_factor = 
                          parameters_neural_network ['scale_factor'], 
                    mode = parameters_neural_network['mode_upsampling'])
        
        
 


#---------------------- Methods Imeplementation -----------------------
      
        
    def forward(self, x):
        
        
        
        print(f'X shape before encoder: {x.shape} \n')
        
        # Forward Propagation in all CNN layers   
        x = self.encoder(x)
        
        
        print(f'X shape after encoder: {x.shape} \n')

        '''
        - Reaching the linear part:
            - need to do global max pooling first
        '''
     
        x = torch.nn.functional.max_pool2d(x,
        kernel_size = (1,x.shape[3]) )
        
        print(f'X shape after maxpooling: {x.shape} \n')
        
        '''
        Since we have input all frames (2 of the past and 2 of the future)
        
        the global max pooling need to be performed on each frame alone,
        
        where each frame have a dimension 1 x something
            - where something is the result coming from the processing
                through the encoder part.
            - that's why we have set the kernel_size to (1,x.shape[3])
            
            where x.shape[3] = width of the feature map
            
            and not (x.shape[2],x.shape[3])
        '''
        

        '''
        Transform the batch format into rank 2 tensor
        to be processed in the fully connected layers
        '''
        x = x.reshape(-1, self._dense_layers_size[0])
        
        print(f'X shape after reshape: {x.shape} \n')

        # Proessing in the dense layers part
        x = self.dense_part(x)
        
        print(f'X shape after dense: {x.shape} \n')
        

        
        # Start Processing through the Decoder
        
        '''
        First we need to adjust again the format of the tensor
        to a rank 4 tensor
        '''

        
        x = \
        x.reshape(self.parameters_neural_network['batch_size'],
                  1,self.parameters_neural_network['size_input'][0],
                   self.parameters_neural_network['dense_layers_list'][-1])
        
        
        print(f'X shape after reshape for decoder: {x.shape} \n')
        
        x = self.decoder(x)
        
        x = self.decoder_restore(x)
        
        print(f'X shape after restored: {x.shape} \n')

        return x
       
 
  
    
       
            
 
