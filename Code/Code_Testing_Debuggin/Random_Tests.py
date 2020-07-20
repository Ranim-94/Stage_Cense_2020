


import Classes_and_Functions.Helper_Functions as hf




saving_location_dict = {
    
    'Directory': 'CreatedDataset/Training_Set_6',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



sensor_dist = hf.count_sensors(saving_location_dict)

labels_map = hf.mapper(sensor_dist)

