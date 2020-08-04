

# *********************** Importing *******************************

import torch

import numpy as np

import os

from math import floor


from Classes_and_Functions.Helper_Functions import mapper,count_sensors




#**************************************************************************

class Dataset_SpecSense(torch.utils.data.Dataset):
    
    
    def __init__(self,saving_location_dict,rows_npy,frame_width,mode):
        
        
        self.saving_location_dict = saving_location_dict
        
        self.mode = mode
        
        self.frame_width , self.rows_npy = frame_width, rows_npy
        
        self.iteartion_per_npy_file = floor(self.rows_npy//self.frame_width)

 

       # -------------- Start Preprocessing  ----------------------- 

       
        # Counting the sensors 
        self.sensor_dist = count_sensors(self.saving_location_dict)
        
        # label mapping: sensor_index <--> some label
        self.label_map = mapper(self.sensor_dist)

        # taking the keys        
        self.sensor_index = tuple(self.label_map.keys())
        

        
        # Filtering all the tiers d'octaves and sensor id
        self.train_spec_list,self.train_id_list = self.__filtering_data()
        
        # Balancing the data
        self.spectr_balanced,self.id_balanced = self.__balancing_data()
        
        
        
        
        # Constructing the data using the balanced data set 
        self.samples , self.samples_id , self.labels  = self.__construct_data()


        
    def __getitem__(self,index):

        
        # Choosing some sample with some index    
        sample_np, label_np = self.samples[:,:,index], self.labels[:,:,index]
            

        if self.mode == 'pretext_task':
            
            
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
            
            
        elif self.mode == 'sensor_classification' or \
        self.mode == 'Fix_Emb' or self.mode == 'Fine_Tune_Emb':
            
            # turning numpy array into pythorch tensors    
            sample = torch.from_numpy(sample_np).float()
            
            label = self.samples_id[index]
            
            # turning into a batch format [volume, height, width]
            sample = sample.unsqueeze(0)
            
  
        return sample,label
    
    
    
    def __len__(self):
        '''
        this method will compute the number of samples we have
        '''  
        
        n_samples = self.iteartion_per_npy_file * len(self.spectr_balanced)
        
        return n_samples
    


    # *********************** Private Methods ************************
    
    
    def __filtering_data(self):
        
        '''
        This function filter the tieres d'octaves and
        the sensor id , and put each in a separate lists
        '''
        
         # Listing all the .npy files in the directory
        list_all_npy_data = os.listdir(self.saving_location_dict['Directory'])
        
         # Filtering: Taking the tiers d'octaves
        self.train_spec_list = [item for item in list_all_npy_data 
            if item.startswith(self.saving_location_dict ['File_Name_Spectrograms'])]
        
        
        # Filtering: taking the sensor id files
        self.id_list = [item for item in list_all_npy_data
         if item.startswith(self.saving_location_dict ['File_Name_sensor_id'])]
        
        
        # Sorting to get a match between the 2 list
        
        '''
        After sorting:
            
            train_spec_50 <---> train_id_50
        '''
        self.train_spec_list.sort()

        self.id_list.sort()
        
        
        return self.train_spec_list,self.id_list
    
    
    def __balancing_data(self):
                  
          # Balancing the different classes:
          
          # Number of balancing for each sensor
          nb_of_samples_balanced = min(self.sensor_dist.values())
          
          name_spectr_convention = \
          self.saving_location_dict['File_Name_Spectrograms']
          
          name_id_convention = \
              self.saving_location_dict['File_Name_sensor_id']
              
                           
          spectr_balanced, id_balanced = [] , []
            
          spectr_by_sensor, id_by_sensor = {} , {}
              
          for id_sensor in self.sensor_index:
              
              '''
              First I will split in a dictionary
              each sensor data
              
              Example: spectr_by_sensor['50'] = list of tiers d'octaves for sensor 50
              
              Take all these data
              
              Then take the required number of samples to construct a 
              balanced data
              '''
              
              # Adding a key with inital empty lists
              spectr_by_sensor[id_sensor] = []
              
              id_by_sensor[id_sensor] = []
              
              for train_spectr,train_id in zip(self.train_spec_list,
                                          self.train_id_list):
                  
                  
                  if train_spectr.startswith(f'{name_spectr_convention}{id_sensor}') \
                  and train_id.startswith(f'{name_id_convention}{id_sensor}'):
                      
                      spectr_by_sensor[id_sensor].append(train_spectr)
                      
                      id_by_sensor[id_sensor].append(train_id)
                      
                
              for i in range(nb_of_samples_balanced):
                  
                  spectr_balanced.append(spectr_by_sensor[id_sensor][i])
              
                  id_balanced.append(id_by_sensor[id_sensor][i])
                  
        
          '''
          spectr_balanced, id_balanced are already
          matched, so no need for sorting
          '''
          
          return spectr_balanced, id_balanced
        
        
    def __construct_data(self):
        
        # Allocating different Numpy arrays to store the data
        
        self.samples = np.empty(  (32,29,len(self.iteartion_per_npy_file 
                                             * self.spectr_balanced)) )
        '''
        Each frame is 125 ms
        32 frames <--> 4 s
            where 16 frames (2s) for future and 16 for past
        '''
        
        # for Audio2Vec reconstruction task
        self.labels = np.empty(  (8,29,
                                  len(self.iteartion_per_npy_file 
                                      * self.spectr_balanced) ) )
        
        '''
        8 frames <--> 1 s window
             In our case, we are trying to reconstrcut 
             1s from 2s in front and 2s in advance
        '''
        
        # for sensor classification
        self.samples_id = np.empty( len(self.iteartion_per_npy_file * self.spectr_balanced)  )
        
        # for storing the data in a 3 dimensional numpy array
        position = 0
        


        # Looping over all the balanced list 
        for count,(spectr,id_sensor) in \
        enumerate(zip(self.spectr_balanced,self.id_balanced)):
            
            # Loading tiers d'octaves
            
            '''
            Chosing some sample in memory mapping mode
            
                - self.saving_location_dict['Directory'] = path of the data 
                
            Memory Mapping allow efficient accessing a file or a portion
            of a file without 
            '''
            
            
            sample_original = \
            np.load(f"{self.saving_location_dict['Directory'] }/{spectr}",
                    mmap_mode = 'r')
            
            # for scanning the single npy file
            start ,end  = 0 , self.frame_width 
            
            
            # Start looping in a single npy file
            for index in range(self.iteartion_per_npy_file):
            
                slice_all = sample_original[start:end,:].copy()
                
                 # taking past slice    
                past_slice = slice_all[:16,:].copy()
                
            
                # taking future slice
                future_slice = slice_all[24:,:].copy()
                
        
                    
                # putting the past and the future to be processed 
                self.samples[:,:,index + position] = np.vstack((past_slice,future_slice))
                
                if self.mode == 'pretext_task':
                
                    # this the middle frame in which we will try to reconstruct
                    # size 8 x 29
                    self.labels[:,:,index + position] = slice_all[16:24,:].copy()
                
                
                elif self.mode == 'sensor_classification':
                
                    # Loading sensor id: these are the true indices of the sensors
                    id_sensor_npy = \
                    np.load(f"{self.saving_location_dict['Directory'] }/{id_sensor}",
                            mmap_mode = 'r')
                    
                    
                    # this will contians the files of the sensor index
                    
                    for key in self.sensor_index:

                        '''
                        Compare the sensor index to the keys I have
                        Once comparison is true I store the label
                        and do a break because I don't need to
                        compare anymore
                        '''
                        
                        if id_sensor_npy[index] == int(key):
                            
                            # Here I store the labels which
                            # start from 0 ---> nb_of_sensors
                            self.samples_id[index + position] = self.label_map[key]
                            

                            break
                        
                # updating the offset  of slicing         
                start = start + self.frame_width
            
                end =  end + self.frame_width
            
            
            
            # update position for storing data after we finish a single npy file
            position += self.iteartion_per_npy_file
        
        
        
        return self.samples , self.samples_id , self.labels 
        
        
        
        
        
        
        
    

        
        
   
    
        
            
        
        
 
        
        
        
        
            
            
            