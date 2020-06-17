


import torch

import numpy as np

import os

import random

import math

class Dataset_SpecSense(torch.utils.data.Dataset):
    
    
    def __init__(self,saving_location_dict,mode):
        
        
        self.saving_location_dict = saving_location_dict
        
        self.mode = mode
        
        '''
        Listing all the .npy files in the directory
        '''
        list_all_npy_data = os.listdir(self.saving_location_dict['Directory'])
        
        
        
        
        '''
        Filtering: Taking the tiers d'octaves
        '''
        self.train_spec_list = [item for item in list_all_npy_data 
            if item.startswith(self.saving_location_dict ['File_Name_Spectrograms'])]
        
        
        # taking the sensor id files
        self.id_list = [item for item in list_all_npy_data
         if item.startswith(self.saving_location_dict ['File_Name_sensor_id'])]
        
        
        # Sorting to get a match between the 2 list
        
        '''
        After sorting:
            
            train_spec_50 <---> train_id_50
        '''
        self.train_spec_list.sort()

        self.id_list.sort()
        
        
        
        
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
            
        start , finish =  3 , sample_original.shape[0] - 2
    
        num1 = random.randint(start,finish)
            
        # taking past slice    
        past_slice = sample_original[num1 - 2: num1,:].copy()
            
        
        # taking future slice
        future_slice = sample_original[num1 + 1 : num1 + 3,:].copy()
        

            
        # putting the past and the future to be processed 
        sample_np = np.concatenate((past_slice,future_slice), axis = 0)
        
        # In case of debuging uncomment these print() statments
        
        # print(f'start: {start} | finish: {finish} | num1: {num1} \n')
        

        # print(f'--> past_slice shape is: {past_slice.shape} \n',
        #       f'--> future_slice shape is: {future_slice.shape}')

        
        # print(f'--> sample_np shape: {sample_np.shape} \n')
        
        

        if self.mode == 'pretext_task':
            
            # this the middle frame in which we will try to reconstruct
            label_np = sample_original[num1,:].copy()
            
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
            
            
        elif self.mode == 'sensor_classification':
            
            
            # this will contians the files of the sensor index
            sample_original_id = np.load(self.saving_location_dict['Directory'] +
                                '/' + self.id_list[index] , mmap_mode = 'r')
            
            '''
            I choose the label 0:4 because 
            we are taking 2 tram from past and 2 from future in each
            spectrogram
            
            It is more compatible to the vectorized implementation
            I have done in the forward propagation for Audi2Vec model
            '''
            label = sample_original_id[0:4].copy()
            
            # turning numpy array into pythorch tensors    
            sample = torch.from_numpy(sample_np).float()
            
            label = torch.from_numpy(label)
            
            # turning into a batch format [volume, height, width]
            sample = sample.unsqueeze(0)
            
  
        return sample,label
    
    
    
    def __len__(self):
        '''
        this method will compute the number of samples we have
        '''  
        
        n_samples = len(self.train_spec_list)
        
        return n_samples
    

        
        
   
    
        
            
        
        
 
        
        
        
        
            
            
            