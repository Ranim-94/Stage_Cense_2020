







'''
Experimenting on Audio2Vec 
'''

import torch




import math


from Classes_and_Functions.Class_Neural_Network_Training import \
Neural_Network_Training

from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Helper_Functions import \
log_CNN_layers,compute_size




'''
--> Creating a dictionary containing different paremeters
for Audio2Vec model
'''

parameters_dic = {
       
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
        
        'batch_size':100
       

       }




saving_location_dict = {
    
    'Directory': 'Training_Set',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



# print('----------- Convolution Size Variation through layers \
#       -------------- \n')

# compute_size(parameters_dic)

# print ('---------------------------- \n')



# Creating the Neural Network instance
net_1 = Audio2Vec(parameters_dic)



optimization_option = {
    
    'neural_network_model':net_1,
    
    'Objective_Function':torch.nn.MSELoss(),
    
    'optimizer':torch.optim.Adam(net_1.parameters(), lr = math.pow(10,-3)),
    
    'epoch_times':1,
    
    'batch_size':parameters_dic['batch_size']
    
    }

set_train = Neural_Network_Training(optimization_option,saving_location_dict)


set_train.training()







# print('----------- Audio2Vec Architecture -------------- \n')

# log_CNN_layers(net_1)


# print ('---------------------------- \n')


