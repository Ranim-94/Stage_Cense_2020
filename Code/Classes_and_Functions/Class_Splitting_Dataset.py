

import shutil

import math

import os

import collections

class Splitting_Datasets:
    
    
    def __init__(self,saving_location_dict,splitting_parameters,
                 splitting_directories_names):
        
        self.__saving_location_dict = saving_location_dict
        
        self.__splitting_parameters = splitting_parameters
        
        self.__splitting_directories_names = splitting_directories_names
        


    def splitt(self):
        
        # Counting all the files in the total data directory and store the names
        # in a list
        list_all_numpy_files = os.listdir(self.__saving_location_dict['Directory'])

        print('--> We have a total of',len(list_all_numpy_files),'.npy files \n')
        
        
        
        '''
        We divide by 3 since we have
            - train_id_xx --> sensor index
            - train_time_stamp_xx --> time stamp measurements
            - train_spec_xx --> sepctrograms for tiers d'octaves
        '''
        total_per_group =  len(list_all_numpy_files)/3 
        
        
        
        size_and_files_dict = collections.OrderedDict()
        
        '''
        Setting the number of splitting:
            - 70 % for training --> how many files we will take
        '''
        
        size_and_files_dict['training_size'] = \
        math.floor(self.__splitting_parameters['training'] * total_per_group)
        
        
        size_and_files_dict['eval_size'] = \
        math.floor(self.__splitting_parameters['eval'] * total_per_group )
        
 
        size_and_files_dict['valid_size' ] = \
            math.floor(self.__splitting_parameters['valid'] * total_per_group)
      
      
        print(size_and_files_dict['training_size'],size_and_files_dict['eval_size'],
              size_and_files_dict['valid_size'])
        
        '''
        Filtering the names to get each data separately in lists
        '''
        size_and_files_dict['all_spec_list'] = [item for item in list_all_numpy_files 
         if item.startswith(self.__saving_location_dict ['File_Name_Spectrograms'])]
        
        
        size_and_files_dict['all_id_list'] = [item for item in list_all_numpy_files
         if item.startswith(self.__saving_location_dict ['File_Name_sensor_id'])]
        
        size_and_files_dict['all_time_stamp_list'] = \
         [item for item in list_all_numpy_files
                if item.startswith(self.__saving_location_dict ['File_Name_time_stamp'])]
      
            
        
        '''
        Sorting so we can  have matching order between all files
        '''
        size_and_files_dict['all_spec_list'].sort()
        
        size_and_files_dict['all_id_list'].sort()
        
        size_and_files_dict['all_time_stamp_list'].sort()
        
        
        '''
        Counting the distribution of the sensors
        '''

           
        '''
        Creating directories
        '''
        
        self.__create_directories()
        
        
        self.__move_files(size_and_files_dict)
        


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
                    
                    
                    
    def __move_files(self,size_and_files_dict):
        
        
        print('--> Start Transferring Training set \n')
  
        for i in range(size_and_files_dict['training_size']):
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_spec_list'][i]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['train_directory_name'])
            
     
        
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_time_stamp_list'][i]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['train_directory_name'])
            
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_id_list'][i]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['train_directory_name'])
            
            
        print('--> Finish Transferring Training set \n')    
        
        print('############################################# \n \n')

        print('--> Start Transferring Eval set \n')        
            
        for i in range(size_and_files_dict['eval_size']):
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_spec_list'] \
            [i + size_and_files_dict['training_size'] ]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['eval_directory_name'])
            
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_time_stamp_list'] \
            [i + size_and_files_dict['training_size']]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['eval_directory_name'])
            
            
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_id_list'] \
            [i + size_and_files_dict['training_size']]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['eval_directory_name'])
            
        print('--> Finish Transferring Eval set \n')    
        
        print('############################################# \n \n')

        print('--> Start Transferring Valid set \n')      
            
            
        for i in range(size_and_files_dict['valid_size']):
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_spec_list'] \
            [i + size_and_files_dict['training_size'] +size_and_files_dict['eval_size'] ]   
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['valid_directory_name'])
            
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_time_stamp_list'] \
            [i + size_and_files_dict['training_size'] + size_and_files_dict['eval_size'] ]      
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['valid_directory_name'])
            
            
            
            source_spec = self.__saving_location_dict['Directory'] + \
            '/' + size_and_files_dict['all_id_list'] \
            [i + size_and_files_dict['training_size'] + size_and_files_dict['eval_size']]      
           
            shutil.move(source_spec,
                        self.__splitting_directories_names['valid_directory_name'])
         
         
        print('--> Finish Transferring Valid set \n')    
        
        print('############################################# \n \n')

       
            
            
            
            
        