


import torch

import numpy as np

import os

import random

import math

class Dataset_SpecSense(torch.utils.data.Dataset):
    
    
    def __init__(self,saving_location_dict,data_percentage):
        
        
        self.saving_location_dict = saving_location_dict
        
        self.__data_percentage = data_percentage
        
        '''
        Listing all the .npy files in the directory
        '''
        list_all_npy_data = os.listdir(self.saving_location_dict['Directory'])
        
        
        
        
        '''
        Filtering: Taking the tiers d'octaves
        '''
        self.train_spec_list = [item for item in list_all_npy_data 
            if item.startswith(self.saving_location_dict ['File_Name_Spectrograms'])]
        

        
    def __getitem__(self,index):
        

        
        '''
        Chosing some sample in memory mapping mode
        
            - self.saving_location_dict['Directory'] = path of the data 
            
        Memory Mapping allow efficient accessing a file or a portion
        of a file without 
        '''
        
        sample_original = np.load(self.saving_location_dict['Directory'] +
                            '/' + self.train_spec_list[index] , mmap_mode = 'r')
        
        
        '''
        Taking the frames of the past and the futures
        '''
        
        start , finish =  3 , sample_original.shape[1] - 2

        num1 = random.randint(start,finish)
        
        # this the middle frame
        label_np = sample_original[num1,:].copy()
        
        
        # the past and the future frames
        sample_np = sample_original[num1 - 2 : num1 + 2,:].copy()
        
            
        # turning numpy array into pythorch tensors    
        sample, label = torch.from_numpy(sample_np).float() , \
        torch.from_numpy(label_np).float()
         
        '''
        The casting .float() is because numpy use float64 as their default type

        We need to cast the tensor to double so the data and the model
        have same data type
        '''
        
        # turning into a batch format [volume, height, width]
        label = label.reshape(-1,29).unsqueeze(0)
        
        sample = sample.unsqueeze(0)
        

            
        return sample,label
    
    
    
    def __len__(self):
        '''
        this method will compute the number of samples we have
        '''  
        
        n_samples = math.floor(self.__data_percentage * math.pow(10,-2) \
                               * len(self.train_spec_list))
        
        return n_samples
 
        
        
        
        
            
            
            