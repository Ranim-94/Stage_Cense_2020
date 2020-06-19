






import torch

import os

import numpy as np

from Classes_and_Functions.Class_Custome_Pytorch_Dataset import \
Dataset_SpecSense



saving_location_dict = {
    
    'Directory': 'CreatedDataset/Training_Set_25',
        
    'File_Name_Spectrograms':'train_spec_',
    
    'File_Name_time_stamp':'train_time_',
    
    'File_Name_sensor_id' : 'train_id_'
    }



# list_all_npy_data = os.listdir(saving_location_dict['Directory'])


# print(f'--> We have {len(list_all_npy_data)} files \n')


# '''
# Filtering: Taking the tiers d'octaves
# '''
# train_spec_list = [item for item in list_all_npy_data 
#             if item.startswith(saving_location_dict ['File_Name_Spectrograms'])]


# print(f'--> We have {len(train_spec_list)} spectrograms \n')

# for count,item in enumerate(train_spec_list) :
    
#     sample_original = np.load(saving_location_dict['Directory'] +
#                                 '/' + item , mmap_mode = 'r')
    
    
#     print(f'sample # {count} shape: {sample_original.shape} \n')



dataset_instance = Dataset_SpecSense(saving_location_dict,mode = 'pretext_task')


# this will give us an iterable object
train_loader = torch.utils.data.DataLoader(dataset = dataset_instance, 
batch_size = 8, shuffle = False)


print(f'--> We have {len(train_loader)} files in train_loader \n')

actual_iter , nb_of_iter = 0, 10000

while actual_iter < nb_of_iter:

    for count,batch in enumerate(train_loader):
    
         # unpacking
         sample , labels = batch 
         
         if sample.shape[2] != 4:
             
             print(f'--> batch # {count} | sample shape: {sample.shape }\n')
             
             
         else:
             
             print(f'--> batch # {count}| Correct Dim')
             
         actual_iter += 1
         
         if actual_iter > nb_of_iter or sample.shape[0] != 10:
             break
     
     
     
     