



from Classes_and_Functions.Class_Custome_Pytorch_Dataset import Dataset_SpecSense

import torch

saving_location_dict = {
    
     'Directory': 'CreatedDataset/Training_Set_50_57',
    
    # 'Directory': '/media/ranim/Seagate Expansion Drive/Training_Set',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }

data_percentage = 25

'''
Loading Data: Instantiate
'''

dataset_instance = Dataset_SpecSense(saving_location_dict,data_percentage)


# trying with some index 



sample, label = dataset_instance[4]

print('--> Sample shape is {} | Sample type: {} \n'.format(sample.shape,type(sample) ) )

      
print('--> Label shape is {} | Label type: {} \n'.format(label.shape,type(label)) )
      
      
      
      



