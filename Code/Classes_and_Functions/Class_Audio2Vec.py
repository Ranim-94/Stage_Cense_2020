

import torch



from Classes_and_Functions.Helper_Functions import conv_block_encoder,\
dense_block, conv_block_decoder


from Classes_and_Functions.Class_Encoder import Encoder


from Classes_and_Functions.Class_Linear_Emb import Linear_Emb

'''
This class implement the Audio2Vec architecture
'''

class Audio2Vec(torch.nn.Module):
       
    def __init__(self, parameters_neural_network):
           
        super().__init__()
      
        self.parameters_neural_network = parameters_neural_network
        
        
        # Instantiate the encoder part
        
        encoder = Encoder(parameters_neural_network)
        
        self.encoder = encoder.enc_block
           
        
       
 #---------------------- Linear and Embedding Part ----------------#

        # Instanting
        lin_emb = Linear_Emb(parameters_neural_network)
                       
        self.dense_part  = lin_emb.dense_part
        
        
        # Needed in forward() method
        self._dense_layers_size = lin_emb._dense_layers_size

 
#------------------------------ Decoder Part ---------------------------

        
        self._CNN_layers_size  = encoder._CNN_layers_size
        
        '''
        Setting the input channel to 1 for the decoder part
        '''

        self._CNN_layers_size[-1] = 1
        
        '''
        Reversing the order of the layers for the Decoder Part
        '''
        self._CNN_layers_size.reverse()
        
        
        decoder_blokcs_list = [conv_block_decoder(in_f, out_f,parameters_neural_network) 
                       for in_f, out_f in zip(self._CNN_layers_size, 
                                              self._CNN_layers_size[1:])]
        
  
        self.decoder = torch.nn.Sequential(*decoder_blokcs_list)
        
        
        self.decoder_restore =  \
        torch.nn.Upsample(scale_factor = 
                          parameters_neural_network ['scale_reconstruction'], 
                    mode = parameters_neural_network['mode_upsampling'])
        
        
 


#---------------------- Methods Imeplementation -----------------------
      
        
    def forward(self, x):
        
        
        
        # print(f'X shape before encoder: {x.shape} \n')
        
        # Forward Propagation in all CNN layers   
        x = self.encoder(x)
        
        
        # print(f'X shape before encoder: {x.shape} \n')

        '''
        - Reaching the linear part:
            - need to do global max pooling first
        '''
     
        x = torch.nn.functional.max_pool2d(x,
        kernel_size = (1,x.shape[3]) )
        
        # print(f'X shape after maxpooling: {x.shape} \n')
        
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
        
        # print(f'X shape after reshape: {x.shape} \n')

        # Proessing in the dense layers part
        x = self.dense_part(x)
        
        # print(f'X shape after dense: {x.shape} \n')
        

        
        # Start Processing through the Decoder
        
        '''
        First we need to adjust again the format of the tensor
        to a rank 4 tensor
        '''

        
        x = \
        x.reshape(self.parameters_neural_network['batch_size'],
                  1,self.parameters_neural_network['size_input'][0],
                  self.parameters_neural_network['dense_layers_list'][-1])
        
        
        # print(f'X shape after reshape for decoder: {x.shape} \n')
        
        x = self.decoder(x)
        
        x = self.decoder_restore(x)
        
        # print(f'X shape after restored: {x.shape} \n')

        return x
       
 
  
    
       
            
 
