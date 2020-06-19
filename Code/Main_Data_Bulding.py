


# ------------------------ Importing -----------------------------------


import math

import os

from collections import OrderedDict

from Classes_and_Functions.Class_Data_Building import SpecCense_Construction

from Classes_and_Functions.Class_Splitting_Dataset_by_sensor import \
Splitting_Datasets_by_sensor

from Classes_and_Functions.Splitting_by_sensor_percentage import \
Splitting_sensor_percentage


#---------------------------------------------------------------------------------

# list containing sensor names
list_sensor_name = os.listdir('Data_Set/')

# Specify different parameter for constructing the total datast

od2 = OrderedDict()

# Sensor index
od2['list_sensor_index'] = [v for v in range(58,60)]

# od2['list_sensor_index'] = [0]

od2['year'] = [2019]

od2['month'] = [12]

od2['days'] = [v for v in range(1,8)] # 1 --> 28


width , margin  = int(math.pow(10,4)) , 250

'''
Casting into int because I use width as index in slicing

- width = 250 ms since the time stamp column in CSV files are in ms

'''


saving_location_dict = {
    
    
    # Specify the name of the directory where the total data (non splitted) is
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
    
    
# Different Data perecentage to be construct
percentage = (6,12,25,50)
    
'''
Choose directories names for the train/eval/valid
'''
splitting_directories_names = {
        
        'train_directory_name': f'CreatedDataset/Training_Set_100', 
        
        'eval_directory_name': f'CreatedDataset/Eval_Set_100',
        
        'valid_directory_name':f'CreatedDataset/Validation_Set_100' 
    
        }



# ****************************************************************************


# **************************** Start Coding ****************************
   


# ***********************************************

# Step 1: Build the total data set (non splitted)

# ***********************************************

# Creating the instance
data_instance = SpecCense_Construction(ordered_dicton_parameters = 
              od2, list_sensor_name = list_sensor_name,     
                width = width, margin = margin, 
              saving_location_dict = saving_location_dict,
              option_remove = False)
    


# Calling the method

data_instance.creating_sample()
 
    
'''
Step 2: we construct train/eval/test for all the data sets

    --> so for 100 % of the available data
'''
# Instantiating   
splitt_worker = Splitting_Datasets_by_sensor(saving_location_dict, 
                  splitting_parameters,splitting_directories_names)
    
'''
Calling the splitt method()
'''
    
splitt_worker.splitt()

# ******************************************************

# Step 3: we constrcut different data %: 6,12,25,50

# ******************************************************




# Creating directories names for different training data precentage
precentage_directories_names = {}

for percent in percentage:
    
    precentage_directories_names[f'train_directory_name_{percent}'] = \
    f'CreatedDataset/Training_Set_{percent}'    
    


'''
Setting the directory now to the total training data constructed
in step 1 so we can constrcut the % data perecentage
'''    
saving_location_dict['Directory'] = splitting_directories_names['train_directory_name']


# Instantiating the class which will construct the different data %
   
percent_builder = Splitting_sensor_percentage(saving_location_dict, 
                  precentage_directories_names,percentage)
    

# Calling the splitt method()    
percent_builder.splitt()
 







 
        
    
    

        
        
        






        
    
    

        
        
        






