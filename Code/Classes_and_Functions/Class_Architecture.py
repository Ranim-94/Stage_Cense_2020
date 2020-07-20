



class Model_Architecture:
    
    
    def __init__(self):
        
        
        self.parameters_Audio2Vec = {
       
           # Input layer

           'size_input': (32,29), # [height, width]
           
            
           # 32: because we have 16 frames (<--> 2s) from past
           # and 16 from future
            
           
           # If we are working with gray scale of RGB images
           'input_volume':1,
           #---------------------------------------
           
           # for CNN layers
           # 'list_filter_nb_per_layer':(64,128,256, 256, 512, 512),
           
           'list_filter_nb_per_layer':(64,128,256),
           
           'padding':1, 'stride': 1,'kernel_size':3,
           
           #---------------------------------------
           
           # for pooling layers
           
           'pooling_option':True,
           
           'padding_pool':0 , 'stride_pool': (1,2) ,'kernel_size_pool':1,
           
           # stride_pool: (for striding height, for striding width)
           
           #---------------------------------------
           
           # for Multilayer perceptrons part: number of neurons
           # for each dense fully connected layer
           'dense_layers_list' : (128,),

           # ---------- Decoder Specification -------------
    
           'mode_upsampling':'nearest', 'scale_factor':(0.79,0.749)
           
           # 'scale_factor':(height,width)
           
           
            
    
           }
        
        
        # for 3 layers: 
        #     scale_factor: (0.79,0.749)
        # for 6 layers:
        #     scale_factor:(0.87,0.838)
            
        
        
        self.parameters_sensor_classification = {
           
           # Input layer

           'size_input': (32,29), # [height, width]
           
           # If we are working with gray scale of RGB images
           
           'volume_input':1 ,
           
           #---------------------------------------
           
           # for CNN layers
           'list_filter_nb_per_layer':(64,128,256),
           
           'padding':1, 'stride': 1,'kernel_size':3,
           
           #---------------------------------------
           
           # for pooling layers
           
           'pooling_option':True,
           
           'padding_pool':0, 'stride_pool': (1,2),'kernel_size_pool':1,
           
           # stride_pool: (for striding height, for striding width)
           
           #---------------------------------------
           
           # for Multilayer perceptrons part: number of neurons
           # for each dense fully connected layer
           'dense_layers_list' : (128,) ,
           
           # nb of classes we are trying to classify: in our case nb of sensors
           'n_classes' : 7
          

           
           }

