


import math

import collections

from Classes_and_Functions.Class_Dataset_Construction \
import SpecCense_Construction


'''
list containing sensor names
'''
list_sensor_name = ['urn:osh:sensor:noisemonitoring:B8-27-EB-EA-EB-EA',
                  'urn:osh:sensor:noisemonitoring:B8-27-EB-EA-12-88']


'''
Creating an ordered dictonary for the necessary paraemeters
in which will change during testing file existence

    - It is ordered since we want to preserve the order of data
    while we are doing iterations and constructing the path
'''

od2 = collections.OrderedDict()


od2['list_sensor_index'] = [0,1]

od2['year'] = [2019]

od2['month'] = [12]

od2['days'] = [v for v in range(1,2)] # 1 --> 28

od2['hour'] = [v for v in range(4)] # 0 --> 23



width , margin  = int(math.pow(10,4)) , 250

'''
Casting into int because I use width as index in slicing
'''

saving_location_dict = {
    
    'Directory': 'Created_Dataset',
    
    'list_sensor_names':list_sensor_name,
    
    'File_Name':'train_spec_'
    }


# Creating the instance
data_instance = SpecCense_Construction(ordered_dicton_parameters = 
                  od2, list_sensor_name = list_sensor_name, \
             width = width, margin = margin, \
             saving_location_dict = saving_location_dict)
    

# Calling the method

data_instance.creating_sample()










