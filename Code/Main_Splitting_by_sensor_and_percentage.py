



from Classes_and_Functions.Splitting_by_sensor_percentage import \
Splitting_sensor_percentage



# *************** Sepcify Different parameters for splitting the data *************


saving_location_dict = {
    
    
    # Specify the name of the directory where the total data is
    'Directory': 'CreatedDataset/Training_Set_50_57',
    
    # Sepcify the names of the files 
    'File_Name_Spectrograms':'train_spec_',
    
    
    # Sepcify the names of the files 
    'File_Name_time_stamp':'train_time_',
    
    
    # Sepcify the names of the files 
    'File_Name_sensor_id' : 'train_id_'
    }


# Different Data perecentage to be tested
percentage = (6,12,25,50)


    


#****************************************************************************


#**************************** Start Coding ****************************
 


'''
Creating directories for different training data precentage
'''

precentage_directories_names = {}

for percent in percentage:
    
    precentage_directories_names[f'train_directory_name_{percent}'] = \
    f'CreatedDataset/Training_Set_{percent}'    
    
'''
Instantiating
'''
    
splitt_instance = Splitting_sensor_percentage(saving_location_dict, 
                  precentage_directories_names,percentage)
    
'''
Calling the splitt method()
'''
    
splitt_instance.splitt()