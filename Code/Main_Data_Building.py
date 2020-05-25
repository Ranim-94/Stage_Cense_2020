


import math

import os

import collections

from Classes_and_Functions.Class_Data_Building \
import SpecCense_Construction




'''
list containing sensor names
'''
list_sensor_name = os.listdir('Data_Set/')




'''
Creating an ordered dictonary for the necessary paraemeters
in which will change during testing file existence

    - It is ordered since we want to preserve the order of data
    while we are doing iterations and constructing the path
'''





od2 = collections.OrderedDict()


'''
Processing:
    - 40 --> 46
'''

od2['list_sensor_index'] = [v for v in range(40,46)]

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
    
    
    # Specify where you want to put your directory
    'Directory': 'CreatedDataset/All_Data',
    
    # Sepcify the names of the files 
    'File_Name_Spectrograms':'train_spec_',
    
    
    # Sepcify the names of the files 
    'File_Name_time_stamp':'train_time_',
    
    
    # Sepcify the names of the files 
    'File_Name_sensor_id' : 'train_id_'
    }


# Creating the instance
data_instance = SpecCense_Construction(ordered_dicton_parameters = 
              od2, list_sensor_name = list_sensor_name,     
                width = width, margin = margin, 
              saving_location_dict = saving_location_dict,
              option_remove = False)
    


# Calling the method

data_instance.creating_sample()










        
        
        






