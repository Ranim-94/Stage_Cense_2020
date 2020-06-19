



from math import pow

from collections import OrderedDict

from Classes_and_Functions.Class_Neural_Network_Training import \
Neural_Network_Training

from Classes_and_Functions.Class_Architecture import Model_Architecture

from Classes_and_Functions.Class_Other_Parameters import Other_Parameters


# Specify which task to train

task = { 'pretext_task': False,
        
        'sensor_classification': True

        }


 
'''
Specifying the prams we want to test

    --> The parameters should be entered in a list 
'''
params_to_try = OrderedDict(
    
    batch_size = [20],

    # percentage of data we need to test
    
    # data_percentage = [6,12,50,100],
    
    data_percentage = [12],

    # rquired nb of iteration ,
    # it is independent of batch size or nb of epoch
    nb_of_iter = [ 1 * int(pow(10,3)) ], 

    shuffle = [False]
    
    )


# ********************* Start Trainining *******************************


# Instantiate secondary parameters
param = Other_Parameters()

# Instantiate the architectures for both models
model = Model_Architecture()

if task['pretext_task'] == True:
    

    coach = Neural_Network_Training(param.optimization_option,model.parameters_Audio2Vec,
                                    param.saving_location_dict,params_to_try,
                                    param.show_trace,param.model_names, 
                                    mode = 'pretext_task')

    coach.training()
    
    
elif task['sensor_classification'] == True:
    
    coach = Neural_Network_Training(param.optimization_option,
                model.parameters_sensor_classification,param.saving_location_dict,
                params_to_try,param.show_trace,param.model_names, 
                mode = 'sensor_classification')

    coach.training()








