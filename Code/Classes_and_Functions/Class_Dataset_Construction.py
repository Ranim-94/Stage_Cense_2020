

import numpy as np

import os

import pandas as pd

import math

import itertools

import shutil 

class SpecCense_Construction:
    
    
    
    def __init__(self,ordered_dicton_parameters,list_sensor_name, \
                 width,margin,saving_location_dict,option_remove):
        

        self.__ordered_dicton_parameters = ordered_dicton_parameters
        
        
        self.__list_sensor_name =  list_sensor_name
        

        self.__width = width
        
        self.__margin = margin
        '''
        maximum time discontinuity allowed between 2 successive 
        frames
        '''
        
        self.__saving_location_dict = saving_location_dict 
        
        self.__option_remove = option_remove
        


    def creating_sample(self):
        
        
        # First Step, Create the dataset directory
        self.__create_dataset_directory()
       

#---------------------------------------------------------------------------
        
        # Second Step: Test file existence
        
        hyperparameters_values = [v for v in \
                            self.__ordered_dicton_parameters.values()]
            
        data_path_list = []
          
        
        hour_string = []
        
        track_sensor = 0
        
        sensor_index_list = self.__ordered_dicton_parameters['list_sensor_index']
        
        
        for sensor_index, year, month , days in \
            itertools.product(*hyperparameters_values):
                
                
     
                '''
                - Checking when we start with another sensor
                    - If yes, displaying a message 
                '''
                
                if track_sensor < len(sensor_index_list) - 1:
                    '''
                    This condition so we don't have out of bound
                    
                    when doing indexing in sensor_index_list via
                    
                    [track_sensor + 1]
                    '''
                
                    if sensor_index == sensor_index_list[track_sensor + 1]:
                        
                        
                            print('************************************** \n')
                        
                            print('--> Moving to sensor #',sensor_index,'\n')
          
                            print('************************************** \n')
                        
                            track_sensor += 1
                
                
                print('- Working with sensor #',sensor_index,'|',  
              'year #',year,'|','month #',month,'|','day #',days,' \n')
                
                data_path_day = 'Data_Set/' + \
                self.__list_sensor_name [sensor_index] + '/' + \
                  str(year) + '/' + str(month) + '/' + str(days) 
                  
                '''
                Listing all the hour.zip file in a particular
                day
                '''
                list_hour_files  = os.listdir(data_path_day)
                
                
                if list_hour_files == []:
                    
                    print('  --> No hour files for day #',days,'\n')
                    
                    
                else:
                    

                    '''
                    1) Inside the day directory, we are listing
                        all the prensent hour.zip 
                        
                    2) splitting the extension .zip from the hour value
                    '''
                    hour_string = [os.path.splitext(item)[0] \
                                   for item in os.listdir(data_path_day)]
                
       
                    '''
                    Transform into a numpy array and sort the hour values
                    becasue os.path.listdr() return unsorted files
                    '''
                    hour_vector = \
                    np.sort(np.array(list(map(int,hour_string)))) 
              
   
                            
                    print('--------------------------- \n')
                            
                    print(' --> Start Slicing the csv file and'\
                         ' creating the .npy data \n')
                                
                    print('--------------------------- \n')
                    
                    
                    '''
                    Constructing the full path with 
                    all hour.zip values
                    '''
                    data_path_list = \
                    [data_path_day +'/' + str(item) +'.zip' for item in hour_vector] 
                            
                    # Create the .npy files
                    self.__slicing(data_path_day,data_path_list,
                                   sensor_index, month,days)
                        
                    print('  --> Finish slicing day # ',days,'\n')
                        
                    print('--------------------------- \n')
                            
                  
                 
    def __create_dataset_directory(self):
        
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
                
            print('- ',self.__saving_location_dict['Directory'],'\n')
            
            os.mkdir(self.__saving_location_dict['Directory'])
     
            print('- Dataset Directory named '+ \
                  self.__saving_location_dict ['Directory'] + \
                  'is created\n')
              
            
                    
    def __slicing(self,data_path_day,
                  data_path_list,sensor_index,month,days):
        
        '''
         Read the csv file using pandas data frame and directly convert to
         numpy array
             
             - header = None in pd.read_csv:
                 -pandas will use auto generated integer values as header
                 - this is need to be specified otherwise it will take
                     the first row as header and skip it while reading
        '''   
        original_numpy_data = \
        np.vstack([pd.read_csv(path,header = None).to_numpy()  
                       for path in data_path_list]) 
            
    
        '''
        NumPy’s vstack stacks arrays in sequence 
        vertically i.e. row wise. 
        And the result is the same as using concatenate with axis=0.
        '''
        
        
        
        if self.__option_remove == True:
            
            '''
            Removing data after saving it
            into the numpy array
            '''
            shutil.rmtree(data_path_day)
             
        
        iteration_per_original_numpy_data = \
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
        for index in range(iteration_per_original_numpy_data):
        
        
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
                        '_'+ str(month) + '_' + str(days) + '_' + str(index) , \
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

            
            # updating the offset  of slicing         
            start = start + self.__width
            
            end =  end + self.__width + 1
            
            '''
            When updating the offset,
            be aware that the end index must be added
            by (width + 1) becasue the end is exclusive in numpy indexing,
            
            otherwise (if we set end = end + width), 
            we will have number of frames equal to (width - 1)
            '''
            
            if index < iteration_per_original_numpy_data - 1:
                
                print('  --> Shifting the slice \n')
    
    
  


       
       
       