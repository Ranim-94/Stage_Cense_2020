


# ------------------------ Importing -----------------------------------


from Classes_and_Functions.Class_Splitting_Dataset_by_sensor import \
Splitting_Datasets_by_sensor



#---------------------------------------------------------------------------------



#*************** Sepcify Different parameters for splitting the data *************

'''
In this part, we construct train/eval/test for all the data sets

so for 100 % of the available data
'''


saving_location_dict = {
    
    
    # Specify the name of the directory where the total data is
    'Directory': 'CreatedDataset/All_Data',
    
    # Sepcify the names of the files 
    'File_Name_Spectrograms':'train_spec_',
    
    
    # Sepcify the names of the files 
    'File_Name_time_stamp':'train_time_',
    
    
    # Sepcify the names of the files 
    'File_Name_sensor_id' : 'train_id_'
    }



'''
Dictionary to specify the percentage of splitting
'''
splitting_parameters = {'training':0.7,'eval':0.2,'valid':0.1}
    
    

# Range of sensor index we are working on
range_sensors_index  = f'50_57'
    
'''
Choose directories names for the train/eval/valid
'''
splitting_directories_names = {
        
        'train_directory_name': f'CreatedDataset/Training_Set_{range_sensors_index}', 
        
        'eval_directory_name': f'CreatedDataset/Eval_Set_{range_sensors_index}',
        
        'valid_directory_name':f'CreatedDataset/Validation_Set_{range_sensors_index}' 
    
        }



#****************************************************************************


#**************************** Start Coding ****************************
    
    
'''
Instantiating
'''
    
splitt_instance = Splitting_Datasets_by_sensor(saving_location_dict, 
                  splitting_parameters,splitting_directories_names)
    
'''
Calling the splitt method()
'''
    
splitt_instance.splitt()



 







 
        
    
    

        
        
        






        
    
    

        
        
        






