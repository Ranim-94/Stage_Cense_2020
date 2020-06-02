







'''
Experimenting on Audio2Vec 
'''

import torch

import math

from collections import OrderedDict

from Classes_and_Functions.Class_Neural_Network_Training import \
Neural_Network_Training



from Classes_and_Functions.Helper_Functions import \
log_CNN_layers,compute_size



# Specifying the prams we want to test
'''
The parameters should be entered in a list 
'''
params_to_try = OrderedDict(
    
    batch_size = [20],
    
    
    # percentage of data we need to test
    
    # data_percentage = [6,12,50,100],
    
    data_percentage = [25,50],

    # rquired nb of iteration ,
    # it is independent of batch size or nb of epoch
    nb_of_iter = [ 2 * int(math.pow(10,1)) ], 
    

    shuffle = [False]
)





'''
--> Creating a dictionary containing different paremeters
for Audio2Vec model
'''

parameters_neural_network = {
       
       # Input layer
       
       # [height, width]
       'size_input': (4,29),
       
       # If we are working with gray scale of RGB images
       'volume_input':1,
       #---------------------------------------
       
       # for CNN layers
       'list_filter_nb_per_layer':(64,128,256,256,512,512),
       
       'padding':1, 'stride': 1,'kernel_size':3,
       
       #---------------------------------------
       
       # for pooling layers
       
       'pooling_option':True,
       
       'padding_pool':0, 'stride_pool': (1,2),'kernel_size_pool':1,
       
       # stride_pool: (for striding height, for striding width)
       
       #---------------------------------------
       
       # for Multilayer perceptrons part: number of neurons
       # for each dense fully connected layer
       'dense_layers_list' : (128,),
       
       
       # ---------- Decoder Specification -------------
       
       
       'mode_upsampling':'nearest', 'scale_factor':(1,0.79),
       
       # 'scale_factor':(height,width)
       
        'scale_reconstruction':(0.3,1),
        
        
        #------------- Batch Size -------------------
        
        # 'batch_size':params_to_try['batch_size']
        
        # # to test Audio2Vec when doing forward propagation
       

       }


'''
/media/ranim/Seagate Expansion Drive/Training_Set
'''

saving_location_dict = {
    
    # 'Directory': 'Training_Set',
    
    'Directory': 'CreatedDataset/Training_Set_50_57',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



# print('----------- Convolution Size Variation through layers \
#       -------------- \n')

# compute_size(parameters_neural_network)

# print ('---------------------------- \n')




optimization_option = {
    
    
    'Objective_Function':torch.nn.MSELoss(),
    
    }




show_trace = True






#********************* Start Trainining *******************************8


set_train = Neural_Network_Training(optimization_option,parameters_neural_network,
                                    saving_location_dict,params_to_try,show_trace)

set_train.training()







# print('----------- Audio2Vec Architecture -------------- \n')

# log_CNN_layers(net_1)


# print ('---------------------------- \n')



