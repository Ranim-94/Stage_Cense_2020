


import shutil

from math import floor

import os


class Splitting_sensor_percentage:
    
    
    def __init__(self,saving_location_dict,precentage_directories_names,
                 percentage):
        
        self.__saving_location_dict = saving_location_dict
                
        self.__precentage_directories_names = precentage_directories_names
        
        self.__percentage = percentage


    def splitt(self):
        
        # Counting all the files in the total data directory and store the names
        # in a list
        list_all_numpy_files = os.listdir(self.__saving_location_dict['Directory'])

       

        '''
        Getting the train_id_xx separately to start counting the distribution
        of the sensors
        '''
        
        sensor_id_list = [item for item in list_all_numpy_files
         if item.startswith(self.__saving_location_dict ['File_Name_sensor_id'])]
        
        
       
        

        '''
        Counting the distribution of the sensors
        '''
        
        count_sensor = self.__count_sensors(sensor_id_list)
        
        
        '''
        Creating directories
        '''
        
        self.__create_directories()
        
        
        for percentage in self.__percentage:
            
            # Copying the files
            self.__copy_files(count_sensor,list_all_numpy_files,
                              percentage)
 
        
        
        

    def __count_sensors(self,sensor_id_list):
        
        
        sliced = len('train_id_xx')
        
 
        # removing the time information month_day_slice_Nb
        for counter,names in enumerate(sensor_id_list) :
            
            sensor_id_list[counter] = \
            names.replace(names,names[:sliced])
        
        
        '''
        Creating an empty dictionary for counting
        '''
        
        count_sensor = {}
        
        '''
        Begin the counting
        '''
        for names in sensor_id_list:
            
            if names in count_sensor:
                
                count_sensor[names] += 1
                
            else:
                
                count_sensor[names] = 1
        
        
        return count_sensor



        
    def __create_directories(self):
            
         
        for key in self.__precentage_directories_names.keys():
            
         
            '''
            Testing if dataset directory exist or not.
            - If not, it is created automatically
            '''
               
            if os.path.isdir(self.__precentage_directories_names[key]):
                        
                print('- Dataset Directory named '+ \
                      self.__precentage_directories_names[key] + ' exists \n')
                
            else:
                        
                '''
                Creating directory
                ''' 
                
                
                os.mkdir(self.__precentage_directories_names[key])
                 
                print('- Dataset Directory named '+ \
                     self.__precentage_directories_names[key] + ' is created\n')
                    
                    
                    
    def __copy_files(self,count_sensor,
                     list_all_numpy_files,percentage):
        
        count_sensor_train_eval_valid = {} # Empty Dictionary
        '''
        Dictionary containing the different splitting numbers we need to split
        for each sensor for different data precentage
          
        Example:
           trrain_id_50 : {'train_nb':72 * 0.25 of the data }
          
        '''
        
        for key in count_sensor.keys():
            
            count_sensor_train_eval_valid[key] = \
            { 'train_nb':floor(count_sensor[key] * percentage * 10**-2 ),
              
        
             } 
                
         
            
        for key,val in count_sensor_train_eval_valid.items():
            
            
            
            # Filtering the first sensor we have for id
            
            '''
            list_all_numpy_files: contains all the sensor names but unordered
            
            We begin to construct a list named 'list_id' to take each sensor alone
            and process them all
            '''
            
            
            '''
            This contains traind_id_xx
            '''
            list_id = [item for item in list_all_numpy_files 
                  if item.startswith(key)]
            
            list_id.sort()
            
            '''
            Now we need to for train_spec_xx and train_time_xx
            
            Methods: I will take the sensor index from key which is train_id_xx
            '''
            sensor_index = key[-2:] # taking xx from train_id_xx by slicing
            
            
            
            # list containg all the spectrogram '.npy' for a particular sensor
            
            list_spectrogram = [item for item in list_all_numpy_files 
           if item.startswith(self.__saving_location_dict['File_Name_Spectrograms'] +
                              str(sensor_index))]
            
            list_spectrogram.sort()
            
            
            # list containg all the time_stamp '.npy' for a particular sensor
            
            list_time_stamp = [item for item in list_all_numpy_files 
           if item.startswith(self.__saving_location_dict['File_Name_time_stamp'] + 
                              str(sensor_index)  )]
            
            
            list_time_stamp.sort()
            
 
            
            '''
            Now we begin Copying to train correspondant directories
            
            '''
            
            # Constructing the training set
            for i in range(val['train_nb']):
                
                '''
                val is the number of files to be copied
                '''
                
                
                # Copying id of sensor
                source = \
                f"{self.__saving_location_dict['Directory']}/{list_id[i]}"
                
                shutil.copy(source,
                self.__precentage_directories_names[f'train_directory_name_{percentage}'])
                
                
                # Copying tiers d'octaves
                source = \
                f"{self.__saving_location_dict['Directory']}/{list_spectrogram[i]}"
                
                shutil.copy(source,
                self.__precentage_directories_names[f'train_directory_name_{percentage}'])
                
                
                # Copying time stamp
                source = \
                f"{self.__saving_location_dict['Directory']}/{list_time_stamp[i]}"
                
                shutil.copy(source,
                self.__precentage_directories_names[f'train_directory_name_{percentage}'])
                
                
           
       
            
            
            
            
        