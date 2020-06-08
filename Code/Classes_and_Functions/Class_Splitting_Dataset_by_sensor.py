


import shutil

import math

import os


class Splitting_Datasets_by_sensor:
    
    
    def __init__(self,saving_location_dict,splitting_parameters,
                 splitting_directories_names):
        
        self.__saving_location_dict = saving_location_dict
        
        self.__splitting_parameters = splitting_parameters
        
        self.__splitting_directories_names = splitting_directories_names
        


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
        Creating directories
        '''
        
        self.__create_directories()
        

        '''
        Counting the distribution of the sensors
        '''
        
        count_sensor = self.__count_sensors(sensor_id_list)
           
 
        # Moving the files
        self.__move_files(count_sensor,list_all_numpy_files)
        
        

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
            
         
        for key in self.__splitting_directories_names.keys():
            
         
            '''
            Testing if dataset directory exist or not.
            - If not, it is created automatically
            '''
               
            if os.path.isdir(self.__splitting_directories_names[key]):
                        
                print('- Dataset Directory named '+ \
                      self.__splitting_directories_names[key] + ' exists \n')
                
            else:
                        
                '''
                Creating directory
                ''' 
                
                
                os.mkdir(self.__splitting_directories_names[key])
                 
                print('- Dataset Directory named '+ \
                     self.__splitting_directories_names[key] + ' is created\n')
                    
                    
                    
    def __move_files(self,count_sensor,list_all_numpy_files):
        
        count_sensor_train_eval_valid = {} # Empty Dictionary
        '''
        Dictionary containing the different splitting numbers we need to split
        for each sensor
          
        Example:
           trrain_id_50 : {'train_nb':72, 'eval_nb':20 , 'valid_nb':10}
          
        '''
        
        for key in count_sensor.keys():
            
            count_sensor_train_eval_valid[key] = \
            { 'train_nb':math.floor(count_sensor[key] * self.__splitting_parameters['training']),
                
              'eval_nb':math.floor(count_sensor[key] * self.__splitting_parameters['eval']),      
                
              'valid_nb':math.floor(count_sensor[key] * self.__splitting_parameters['valid'])
        
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
            
            Methods: I will take the sesor index from key which is train_id_xx
            '''
            sensor_index = key[-2:] # taking xx from train_id_xx by slicing
            
            
            
            # list containg all the spectrogram '.npy' for a particular sensor
            list_spectrogram = [item for item in list_all_numpy_files 
           if item.startswith(self.__saving_location_dict['File_Name_Spectrograms']+ str(sensor_index))]
            
            list_spectrogram.sort()
            
            
            # list containg all the time_stamp '.npy' for a particular sensor
            list_time_stamp = [item for item in list_all_numpy_files 
           if item.startswith(self.__saving_location_dict['File_Name_time_stamp'] + str(sensor_index)  )]
            
            
            list_time_stamp.sort()
            
 
            
            '''
            Now we begin moving to train/eval/valid correspondant directories
            
            '''
            
            # Constructing the training set
            for i in range(val['train_nb']):
                
                '''
                val is another dictionary
                '''
        
                source = f"{self.__saving_location_dict['Directory']}/{list_id[i]}"
                
                shutil.move(source,self.__splitting_directories_names['train_directory_name'])
                
                
                source = f"{self.__saving_location_dict['Directory']}/{list_spectrogram[i]}"
                
                shutil.move(source,self.__splitting_directories_names['train_directory_name'])
                
                
                source = f"{self.__saving_location_dict['Directory']}/{list_time_stamp[i]}"
                
                shutil.move(source,self.__splitting_directories_names['train_directory_name'])
                
                
                
              # Constructing the eval set
            for i in range(val['eval_nb']):
                
                '''
                val is another dictionary
                '''
                
                # Moving sensor id
                source = \
                f"{self.__saving_location_dict['Directory']}/{list_id[ i + val['train_nb'] ] }"
                
                shutil.move(source,self.__splitting_directories_names['eval_directory_name'])
                
                # Moving tiers d'octaves
                source = \
                f"{self.__saving_location_dict['Directory']}/{list_spectrogram[i + val['train_nb']]}"
                
                shutil.move(source,self.__splitting_directories_names['eval_directory_name'])
                
                
                # Moving time stamp
                source = \
                f"{self.__saving_location_dict['Directory']}/{list_time_stamp[i + val['train_nb']]}"
                
                shutil.move(source,self.__splitting_directories_names['eval_directory_name'])
                
                
                
                
              # Constructing the valid set
            for i in range(val['valid_nb']):
                
                '''
                val is another dictionary
                '''
                
                # Moving sensor id
                source = \
                f"""{self.__saving_location_dict['Directory']}/{list_id[i + val['train_nb'] + 
                val['eval_nb'] ] }"""
                
                shutil.move(source,self.__splitting_directories_names['valid_directory_name'])
                
                # Moving tiers d'octaves
                source = \
                f"""{self.__saving_location_dict['Directory']}/{list_spectrogram[ i + 
                val['train_nb'] + val['eval_nb'] ]}"""
                
                shutil.move(source,self.__splitting_directories_names['valid_directory_name'])
                
                
                # Moving time stamp
                source = \
                f"""{self.__saving_location_dict['Directory']}/{list_time_stamp[i + 
                val['train_nb'] + val['eval_nb'] ]}"""
                
                shutil.move(source,self.__splitting_directories_names['valid_directory_name'])

       
            
            
            
            
        