


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
        Chosing some sample and directly turn it into a Pytorch Tensor
        '''
        sample_original = \
        torch.from_numpy(np.load(self.saving_location_dict['Directory'] +
                                 '/' + self.train_spec_list[index])).float()
        
        '''
        The casting .float() is because numpy use float64 as their default type

        We need to cast the tensor to double so the data and the model
        have same data type
        '''
        
        
        '''
        Taking the frames of the past and the futures
        '''
        
        start , finish =  3 , sample_original.shape[1] - 2

        num1 = random.randint(start,finish)
        
        
        label = sample_original[num1,:].clone().reshape(-1,29).unsqueeze(0)
        
        sample = sample_original[num1 - 2 : 
                          num1 + 2,:].clone().unsqueeze(0)
            
            
        return sample,label
    
    
    
    def __len__(self):
        '''
        this method will compute the number of samples we have
        '''  
        
        n_samples = math.floor(self.__data_percentage * math.pow(10,-2) * len(self.train_spec_list))
        
        return n_samples
 
        
        
        
        
            
            
            