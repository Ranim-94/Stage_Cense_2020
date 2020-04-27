


import math

import collections

from Classes_and_Functions.Class_Dataset_Construction \
import SpecCense_Construction

import os


'''
list containing sensor names
'''
# list_sensor_name = ['urn:osh:sensor:noisemonitoring:B8-27-EB-1A-0B-6C',
#                   'urn:osh:sensor:noisemonitoring:B8-27-EB-1F-AB-9F',
#                   'urn:osh:sensor:noisemonitoring:B8-27-EB-03-5C-6B',
#                   'urn:osh:sensor:noisemonitoring:B8-27-EB-3B-82-1C',
#                   'urn:osh:sensor:noisemonitoring:B8-27-EB-4B-F1-E1',
#                   'urn:osh:sensor:noisemonitoring:B8-27-EB-4D-21-0D']



list_sensor_name = os.listdir('Data_Set/')


'''
Creating an ordered dictonary for the necessary paraemeters
in which will change during testing file existence

    - It is ordered since we want to preserve the order of data
    while we are doing iterations and constructing the path
'''

od2 = collections.OrderedDict()


od2['list_sensor_index'] = [v for v in range(0,1)]

od2['year'] = [2019]

od2['month'] = [12]

od2['days'] = [v for v in range(1,29)] # 1 --> 28

# od2['hour'] = [v for v in range(3)] # 0 --> 23



width , margin  = int(math.pow(10,4)) , 250

'''
Casting into int because I use width as index in slicing
'''

saving_location_dict = {
    
    'Directory': 'Created_Dataset',
        
    'File_Name':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }


# Creating the instance
data_instance = SpecCense_Construction(ordered_dicton_parameters = 
                  od2, list_sensor_name = list_sensor_name, \
             width = width, margin = margin, \
             saving_location_dict = saving_location_dict)
    

# Calling the method

data_instance.creating_sample()










