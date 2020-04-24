

import numpy as np

import os

import pandas as pd

import math

import itertools

class SpecCense_Construction:
    
    
    
    def __init__(self,ordered_dicton_parameters,list_sensor_name, \
                 width,margin,saving_location_dict):
        

        self.__ordered_dicton_parameters = ordered_dicton_parameters
        
        
        self.__list_sensor_name =  list_sensor_name
        

        self.__width = width
        
        self.__margin = margin
        '''
        maximum time discontinuity allowed between 2 successive 
        frames
        '''
        
        self.__saving_location_dict = saving_location_dict 
        


    def creating_sample(self):
        
        
        # First Step, Create the dataset directory
        
        '''
        Testing if dataset directory exist or not.
            - If not, it is created automatically
        '''
   
        if os.path.isdir(self.__saving_location_dict ['Directory']):
            
            print('- Dataset Directory named '+ \
                  self.__saving_location_dict ['Directory']+ ' exists \n')
            
                
        else:
            
            '''
            Creating directory
            '''
                
            os.makedirs(self.__saving_location_dict['Directory'])
     
            print('- Dataset Directory named '+ \
                  self.__saving_location_dict ['Directory'] + \
                  'is created\n')
            
            
            
            
        
#---------------------------------------------------------------------------
        
        # Second Step: Test file existence
        
        hyperparameters_values = [v for v in \
                            self.__ordered_dicton_parameters.values()]
            
        data_path = []
          
        hours_vector = np.empty(4) # testing for 4 hours
        
        counter_files = 0
        
        for sensor_index, year, month , days , hour in \
            itertools.product(*hyperparameters_values):
                
                
                
                
                print('- Working with sensor #',sensor_index,'|',  
              'year #',year,'|','month #',month,'|','day #',days,'|', 
              'hour #',hour,'| \n')
                
                data_path.append('Data_Set/' + \
                self.__list_sensor_name [sensor_index] + '/' + \
                  str(year) + '/' + str(month) + '/' + \
                  str(days) + '/' + str(hour) + '.zip') 
                  
                hours_vector[hour] = hour
                  
                print('  --> Testing file existence:', end = '')
                      
                if os.path.isfile(data_path[hour]) == True:
                    
                    '''
                    Here we create the data set
                    '''
                    # increment the counter_file by 1
                    counter_files += 1 
                    
                    print(' File Exist, we have,',counter_files,
                          'files \n')
                    
                    
                    if counter_files == hours_vector.shape[0]:
                        '''
                        Ensure that we have 4 csv files
                        '''
                        
                        hour_diff = np.diff(hours_vector)
                        
                        test_hour_shape = hour_diff[hour_diff > 1].shape
                        
                        if test_hour_shape[0] == 0:
                            
                            '''
                            Ensure that the 4 csv files are consecutive in
                            hours
                                - In other words, time difference 
                                is not > 1 hour
                            '''
                            
                            print('  --> All the', counter_files,
                                  ' files are consecutive in hours \n')
                            
                            print('--------------------------- \n')
                            
                            print(' --> Start Slicing the csv file and'\
                                  ' creating the .npy data \n')
                                
                            print('--------------------------- \n')
                            
                            # Create the .npy files
                            self.__slicing(data_path,sensor_index , month,
                                   days)
                        
                    
                else:
                    
                    print(' No File \n')
                 
        
        # Clearing the content for next testing
        data_path.clear()
                
        counter_files = 0
                
                
                
                
                    
    def __slicing(self,data_path,sensor_index,month,days):
        
        '''
         Read the csv file using pandas data frame and directly convert to
         numpy array
             
             - header = None in pd.read_csv:
                 -pandas will use auto generated integer values as header
                 - this is need to be specified otherwise it will take
                     the first row as header and skip it while reading
        '''   
        original_data_list  = []
            
        for i in range(len(data_path)):
            
            original_data_list.append(pd.read_csv(data_path[i],
                                                  header = None).to_numpy()) 
            
            
        original_numpy_data = np.vstack(original_data_list)
        
        
        iteration_per_csv_file = \
        math.floor(original_numpy_data.shape[0]/ self.__width) 
        
        '''
        math.floor: will round down to the nearest integer
     
        ''' 
    
        start, end  = 0 , self.__width
    
        '''
        end = width and not (width - 1) because the slcing in numpy is exclusive
        
        - so if width = 100 (we want 100 frames)
            [0:100] will be from 0 --> 99, which are 100 frames
    
        '''
        
        
        
        '''
        Constructing the path components:
        
        Component in wich where we store the .npy data
        
        It will be used in np.save()
        '''
        
        # For octave spectrograms
        filename = \
                os.path.join(self.__saving_location_dict ['Directory'],
               self.__saving_location_dict ['File_Name'])
                
           
        # For time stamp info
        filename_time_stamp = \
                os.path.join(self.__saving_location_dict ['Directory'],
               self.__saving_location_dict ['File_Name_time_stamp'])
                
                
         # For time stamp info
        filename_sensor_id = \
                os.path.join(self.__saving_location_dict ['Directory'],
               self.__saving_location_dict ['File_Name_sensor_id'])
                
        
        sensor_id_vec = np.full(self.__width, sensor_index)
        
        '''
        np.full() create an array filled with values = sensor_index, 
        of length  = self.__width
        '''
       
                
        # start slicing through the csv file
        for index in range(iteration_per_csv_file):
        
        
            '''
            taking the unix tim stamp column
            '''
            
            unix_time_stamp_measured = \
            original_numpy_data[start:end,0].copy()
            
            '''
            Computing the difference 
            '''
            diff = np.diff(unix_time_stamp_measured)
            
            
            '''
            Checking for frame continuity
            '''
            
            test_shape = diff[diff > self.__margin].shape
            
            print('  --> Checking frame continuity '
                  'for slice # ' ,index, end = '')
    
            if test_shape[0] == 0:
                
                
                print(' : Frames are continous \n')
        
                '''
                - All the frames are continous
                    -No time stampe above the margin
            
                - Here we create the .npy files
                '''
                
                 # saving the sensor id
                
                np.save(filename_sensor_id + str(sensor_index) + \
                        '_'+ str(month) + '_' + str(days) + '_' \
                         + str(index) , sensor_id_vec)
                
                # saving the time stamp
                
                np.save(filename_time_stamp + str(sensor_index) + \
                        '_'+ str(month) +'_' + str(days) + '_'  \
                         + str(index) , unix_time_stamp_measured)
                
                 
                # saving the ocataves spectrograms                 
                np.save(filename + str(sensor_index) + \
                        '_'+ str(month) + str(days) + '_' + str(index) , \
                        original_numpy_data[start:end, 3:])
                    
                '''
                 - the 3 index in [start:end, 3:] 
                 is where the octave bands starts in the csv file
                '''
        
                print('  --> Finish creating .npy files \n')
        
            else:
                
                '''
                - There is discontinuity
                    - Reject the frame                
                '''            
                
                print(' : Theres is discontinuity, \
                      slice is rejected \n')    

            
            # updating the offset         
            start = start + self.__width
            
            end =  end + self.__width + 1
            
            '''
            When updating the offset,
            be aware that the end index must be added
            by (width + 1) becasue the end is exclusive in numpy indexing,
            
            otherwise (if we set end = end + width), 
            we will have number of frames equal to (width - 1)
            '''
            
            print('  --> Shifting the slice \n')
    
    
  


       
       
       