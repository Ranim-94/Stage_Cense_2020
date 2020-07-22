



from math import pow

from collections import OrderedDict



from Classes_and_Functions.Class_Neural_Network_Training_Valid import \
Neural_Network_Training_Valid

from Classes_and_Functions.Class_Architecture import Model_Architecture

from Classes_and_Functions.Class_Other_Parameters import Other_Parameters

from Classes_and_Functions.Helper_Functions import plot_results

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
    
    data_percentage = [6],

    # rquired nb of iteration ,
    # it is independent of batch size or nb of epoch
    nb_of_iter = [ 700 ], 

    shuffle = [True]
    
    )



save_point, start_from_iter, resume_training = \
350 , 0 , False


loaded_model = 'Saved_Iteration/Audio2Vec_emb_70000_6.pth'

# ********************* Start Trainining *******************************


# Instantiate secondary parameters
param = Other_Parameters()

# Instantiate the architectures for both models
model = Model_Architecture()

if task['pretext_task'] == True:
    

    coach = Neural_Network_Training_Valid(param.optimization_option,
                                    model.parameters_Audio2Vec,
                                    param.saving_location_dict,params_to_try,
                                    param.frame_width, param.rows_npy,
                                    param.show_trace,param.model_names,save_point, 
                                    start_from_iter,resume_training,
                                    loaded_model,mode = 'pretext_task')

    _,accuracy_validation = coach.training()
    
    
    iter_nb = int(params_to_try['nb_of_iter'][0]) 
    
    percentage = params_to_try['data_percentage'][0]
    
    
    # param_plot = {
    
    # # Enter the name of the saved pth file
    # 'name_file': f"Audio2Vec_emb_100000_6.pth",
    
    # 'title':f"Audio2Vec Using Satble Adam and Scheduler",
    
    # 'xlabel':'Iteration Number',
    
    # 'ylabel':'Mean Square Error',
    
    # 'save':f"Audio2Vec_emb_{iter_nb}_{percentage}.pdf"
    
    
    # }


    # plot_results(param_plot,resume_training, start_from_iter ,iter_nb)
    
    
elif task['sensor_classification'] == True:
    
    coach = Neural_Network_Training_Valid(param.optimization_option,
                model.parameters_sensor_classification,
                param.saving_location_dict,
                params_to_try,
                param.frame_width, param.rows_npy,
                param.show_trace,param.model_names,save_point, 
                start_from_iter,resume_training,
                loaded_model,mode = 'sensor_classification')

    valid_loss_per_epoch,accuracy_validation = coach.training()


    # iter_nb = int(params_to_try['nb_of_iter'][0])
    
    # percentage = params_to_try['data_percentage'][0]

    # param_plot = {
    
    
    # 'name_file': f"classif_no_emb_60000_6_Stable_Adam.pth",
    
    # 'title':f"Task: Sensor Classification Using Adam ",
    
    # 'xlabel':'Iteration Number',
    
    # 'ylabel':'Cross Entropy',
    
    # 'label':'Stable_Adam',
    
    # 'save':f"Classif_no_emb_{iter_nb}_{percentage}_Satble_Adam.pdf"
    
    
    # }


    # plot_results(param_plot,resume_training, start_from_iter ,iter_nb)







