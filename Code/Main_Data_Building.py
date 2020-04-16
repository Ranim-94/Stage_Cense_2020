


import pandas as pd

import numpy as np

import torch


import math

from Classes_and_Functions.Class_SpecCense_Dataset import \
SpecCense_Dataset 







'''
Constructing a dictionary of the data path
containing neccessary parameters
'''

path_dictionary = {


    'sensor_index':0,
    
    'sensor_name' : ['urn:osh:sensor:noisemonitoring:B8-27-EB-EA-EB-EA',
                    'urn:osh:sensor:noisemonitoring:B8-27-EB-EA-12-88'],
    
     # this is fixed for now for all sensors
    'year' : str(2019), # this is fixed
    
    'month_option' : str(12) , # this is fixed,

    'day_option' : str(24) ,

    'hour_option' : str(3) ,
        
}




path_2 = 'Data_Set/' + path_dictionary['sensor_name'][path_dictionary ['sensor_index']] + '/' + \
path_dictionary['year'] + '/' + path_dictionary['month_option'] + '/' + \
path_dictionary ['day_option'] + '/' + path_dictionary['hour_option'] + '.zip'



'''
 Read the csv file using pandas data frame and directly convert to
 numpy array
     
     - header = None in pd.read_csv:
         -pandas will use auto generated integer values as header
         - this is need to be specified otherwise it will take
             the first row as header and skip it while reading

'''
sensor_numpy = pd.read_csv(path_2, header = None).to_numpy()



'''
print the type and shape to be sure 
'''
print('- Type of sensor_numpy:',type(sensor_numpy),'\n')

print('- Sensor_numpy shape is:',sensor_numpy.shape,'\n')


nb_window_frame_per_csv_file,_ = sensor_numpy.shape

#width = int(math.pow(10,3)) 


width = 1000

iteration_per_csv_file = math.floor(nb_window_frame_per_csv_file/width)




'''
math.floor: will round down to the nearest integer
 
 
'''



# Creating the instance
data_instance = SpecCense_Dataset(sensor_numpy, width,iteration_per_csv_file )

# Testing the attribute 


print('- x_data type:',type(data_instance.x_data),'\n')

print('- data_instance shape:',data_instance.x_data.shape,'\n')


# testing getitem
first_sample = data_instance[0]

print('- first_sample shape is:',first_sample.shape,'\n')



# testing len
print('- Number of training examples is:',len(data_instance),'\n')



# testing dataloader 

batch = 4

# this will give us an iterable object
train_loader = torch.utils.data.DataLoader(dataset = data_instance, 
                          batch_size = batch,
                          shuffle = True)


for counter_loop , batch in enumerate(train_loader):
         
    
    print('* Sample number #',counter_loop,': \n \n',
          '- features shape:',batch.shape)




