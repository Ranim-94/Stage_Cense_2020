




import torch

from Classes_and_Functions.Class_Encoder import Encoder

from Classes_and_Functions.Class_Linear_Emb import Linear_Emb


'''
This class implement the Audio2Vec architecture
'''

class My_Calssifier_Encoder(torch.nn.Module):
       
    def __init__(self, parameters_neural_network):
           
        super().__init__()
      
        self.parameters_neural_network = parameters_neural_network
        
      
         # Instantiate the encoder part
        
        encoder = Encoder(parameters_neural_network)
        
        self.encoder = encoder.enc_block
       
 #---------------------- Linear and Embedding Part ----------------#

    
                       
        # Instanting the Liner_Emb class
        lin_emb = Linear_Emb(parameters_neural_network)
                       
        self.dense_part  = lin_emb.dense_part
        
        # Needed in forward() method
        self._dense_layers_size = lin_emb._dense_layers_size
        
# ----------------- Output Layer: Classification Part ----------------------       
        
        
        self._dense_layers_size = lin_emb._dense_layers_size
        
        self.last = torch.nn.Linear(self._dense_layers_size[-1], 
                              self.parameters_neural_network['n_classes']) 

 

 


#---------------------- Methods Imeplementation -----------------------
      
        
    def forward(self, x):
  
        '''
        In case you want to test forward propagation
        method, uncomment the print() statements
        '''
        
        # print(f'--> X shape before encoder is: {x.shape} \n')
        
        # Forward Propagation in all CNN layers   
        x = self.encoder(x)
        
        # print(f'--> X shape after encoder is: {x.shape} \n')

        '''
        - Reaching the linear part:
            - need to do global max pooling first
        '''
     
        x = torch.nn.functional.max_pool2d(x,
        kernel_size = (1,x.shape[3]) )
        
        
        # print(f'--> X shape after maxpooling is: {x.shape} \n')
        
        
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
        
        '''
        Important Note:
        ----------------    
        
        In the case of Sensor classification, the reshaping
        for processing through the dense part is different than
        pretext task in spectrogram reconstruction.
        
        
        Here, we only want the proababilty distribution of the sensors
        
        which is encoded by the batch size, we don't want to input
        all the frame for reconstuction as we did in the pretext task
        '''
        
        
        
        # print(f'--> X shape after reshaping is: {x.shape} \n')
        

        # Proessing in the dense layers part
        x = self.dense_part(x)
        
        # print(f'--> X shape after dense is: {x.shape} \n')
        

        # Passing throught the output layer
        x = self.last(x)
        
        '''
        The shape of x will be: nb of batches x nb of sensors 
        to be classified
        
        Example: x.shape = torch.Size([80,7])
        
        where: we have 80  frames (so 20 x 4)
        
        and 7 is the probability distribution of the sensors produced
        later by softmax
        
        '''
        
        
        # print(f'--> X shape after output layer is: {x.shape} \n')
   
        return x
 
       
 
  
    
       
            
 
