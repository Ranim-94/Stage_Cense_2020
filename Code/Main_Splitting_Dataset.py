



import os

import math

import shutil

list_sensor_name = os.listdir('CreatedDataset')



training_size = math.floor(0.7 * ( len(list_sensor_name)/3) )  


saving_location_dict = {
    
    'Directory': 'CreatedDataset',
        
    'File_Name':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }





train_spec_list = [item for item in list_sensor_name 
 if item.startswith(saving_location_dict ['File_Name'])]


train_id_list = [item for item in list_sensor_name
 if item.startswith(saving_location_dict ['File_Name_sensor_id'])]

train_time_list = [item for item in list_sensor_name
        if item.startswith(saving_location_dict ['File_Name_time_stamp'])]


       
train_spec_list.sort()

train_id_list.sort()

train_time_list.sort()   


# for i in range(training_size):
    
#     source_spec = 'CreatedDataset/'+ train_spec_list[0]   
   
#     shutil.move(source_spec,'train')
    
    


        
        
        