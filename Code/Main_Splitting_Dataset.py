




from Classes_and_Functions.Class_Splitting_Dataset \
import Splitting_Datasets

saving_location_dict = {
    
    'Directory': 'CreatedDataset',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }


splitting_parameters = {
    
    'training':0.7,
    
    'eval':0.2,
    
    'valid':0.1,
    
    }


splitting_directories_names = {
    
    'train_directory_name':'Training_Set',
    
    'eval_directory_name':'Eval_Set',
    
    'valid_directory_name':'Validation_Set',
    
    }


'''
Instantiating
'''

splitt_instance = \
Splitting_Datasets(saving_location_dict, splitting_parameters, 
                                     splitting_directories_names)

'''
Calling the splitt method()
'''

splitt_instance.splitt()







        
        
        