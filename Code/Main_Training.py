





from collections import OrderedDict

from Classes_and_Functions.Class_Neural_Network_Training_Valid import Neural_Network_Training_Valid

from Classes_and_Functions.Class_Architecture import Model_Architecture

from Classes_and_Functions.Class_Other_Parameters import Other_Parameters

from Classes_and_Functions.Helper_Functions import plot_results

# Specify which task to train

task = { 'pretext_task': False,
        
        'sensor_classification': False,
        
        'Fix_Emb': True,
        
        'Fine_Tune_Emb': False

        }


 
'''
Specifying the prams we want to test

    --> The parameters should be entered in a list 
'''
params_to_try = OrderedDict(
    
    batch_size = [20],

    # percentage of data we need to test
    
    # data_percentage = [6,12,50,100],
    
    data_percentage = [6],

    # rquired nb of iteration ,
    # it is independent of batch size or nb of epoch
    nb_of_iter = [ 6*10**4 ], 

    shuffle = [True]
    
    )



start_from_iter, resume_training = 0 , False



loaded_model = 'Saved_Iteration/Audio2Vec_emb_70000_6.pth'

# ********************* Start Trainining *******************************


# Instantiate secondary parameters
param = Other_Parameters()

# Instantiate the architectures for both models
model = Model_Architecture()

if task['pretext_task'] == True:
    

    coach = Neural_Network_Training_Valid(param.optimization_option,
                                    model,
                                    param.saving_location_dict,params_to_try,
                                    param.frame_width, param.rows_npy,
                                    param.show_trace,param.model_names, 
                                    start_from_iter,resume_training,
                                    loaded_model,mode = 'pretext_task')

    valid_loss_per_epoch,_ = coach.training()
    
    
   
    
    
elif task['sensor_classification'] == True:
    
    coach = Neural_Network_Training_Valid(param.optimization_option,
                model,
                param.saving_location_dict,
                params_to_try,
                param.frame_width, param.rows_npy,
                param.show_trace,param.model_names, 
                start_from_iter,resume_training,
                loaded_model,mode = 'sensor_classification')

    _,accuracy_validation = coach.training()


    
elif task['Fix_Emb'] == True:
    
    coach = Neural_Network_Training_Valid(param.optimization_option,
                model,
                param.saving_location_dict,
                params_to_try,
                param.frame_width, param.rows_npy,
                param.show_trace,param.model_names, 
                start_from_iter,resume_training,
                loaded_model,mode = 'Fix_Emb')

    _,accuracy_validation = coach.training()
    
    
elif task['Fine_Tune_Emb'] == True:
    
    coach = Neural_Network_Training_Valid(param.optimization_option,
                model,
                param.saving_location_dict,
                params_to_try,
                param.frame_width, param.rows_npy,
                param.show_trace,param.model_names, 
                start_from_iter,resume_training,
                loaded_model,mode = 'Fine_Tune_Emb')

    _,accuracy_validation = coach.training()






