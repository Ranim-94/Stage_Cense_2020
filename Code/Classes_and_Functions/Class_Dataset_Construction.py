

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
            
            
        for sensor_index, year, month , days , hour in \
            itertools.product(*hyperparameters_values):
                
                print('- Working with sensor #',sensor_index,'|',  
              'year #',year,'|','month #',month,'|','day #',days,'|', 
              'hour #',hour,'| \n')
                
                data_path = 'Data_Set/' + \
                self.__list_sensor_name [sensor_index] + '/' + \
                  str(year) + '/' + str(month) + '/' + \
                  str(days) + '/' + str(hour) + '.zip'
                  
                  
                print('  --> Testing file existence:', end = '')
                      
                if os.path.isfile(data_path) == True:
                    
                    '''
                    Here we create the data set
                    
                    '''
                    
                    print(' File Exist \n')
                    
                    # Create the .npy files
                    self.__slicing(data_path,sensor_index,days,hour)
                    
                    
                else:
                    
                    print(' No File \n')
                    
                    
    def __slicing(self,data_path,sensor_index,days,hour):
        
        '''
         Read the csv file using pandas data frame and directly convert to
         numpy array
             
             - header = None in pd.read_csv:
                 -pandas will use auto generated integer values as header
                 - this is need to be specified otherwise it will take
                     the first row as header and skip it while reading
        '''   
        original_numpy_data  = \
        pd.read_csv(data_path,header = None).to_numpy()
        
        iteration_per_csv_file = \
        math.floor(original_numpy_data.shape[0]/ self.__width) 
        
        '''
        math.floor: will round down to the nearest integer
     
        '''
        
    
        start = 0
    
        end = self.__width 
    
        '''
        end = width and not width -1 because the slcing in numpy is exclusive
        
        - so if width = 100 (we want 100 frames)
            [0:100] will be from 0 -->99 which are 100 frames
    
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
        np.full create an array filled with sensor_index of length
        
        self.__width
        '''
       
                
        # start slicing through the csv file
        for index in range(iteration_per_csv_file):
        
        
            '''
            taking the unix tim stamp column
            '''
            
            unix_time_stamp_measured = original_numpy_data[start:end,0].copy()
            
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
                -All the frames are continous
                    -No time stampe above the margin
            
                Here we create the .npy files
                '''
                
                 # saving the sensor id
                
                np.save(filename_sensor_id + str(sensor_index) + \
                        '_' + str(days) + '_' + str(hour) + \
                        '_' + str(index) , sensor_id_vec)
                
                # saving the time stamp
                
                np.save(filename_time_stamp + str(sensor_index) + \
                        '_' + str(days) + '_' + str(hour) + \
                        '_' + str(index) , unix_time_stamp_measured)
                
                 
                # saving the ocataves spectrograms                 
                np.save(filename + str(sensor_index) +  '_' + \
                        str(days) + '_' + str(hour) + '_' + str(index) , \
                        original_numpy_data[start:end, 3:])
        
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
            
            end =  end + self.__width
            
            print('  --> Shifting the slice \n')
    
    
    
    
    '''
    - Description of the function:    
    
    -At a first step we matrix of shape 29 x width
        - where the slicing [start:end] will always
            be of length = width
        
        - the 3 index in [start:end, 3:] 
        is where the octave bands starts in the csv file
            - we have 29 octave bands we are working on
                that's why the shape will be 29 x width
    '''





       
       
       