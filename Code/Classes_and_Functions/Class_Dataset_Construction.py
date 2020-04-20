









import numpy as np

import os

import pandas as pd

import math

class SpecCense_Construction:
    
    
    
    def __init__(self,data_path,width,margin,saving_location_dict):
        
        
        '''
         Read the csv file using pandas data frame and directly convert to
         numpy array
             
             - header = None in pd.read_csv:
                 -pandas will use auto generated integer values as header
                 - this is need to be specified otherwise it will take
                     the first row as header and skip it while reading

        '''
        
        self.__original_numpy_data  = pd.read_csv(data_path,header = None).to_numpy()
        
        self.__width = width
        
        self.__margin = margin
        
        self.__iteration_per_csv_file = \
        math.floor(self.__original_numpy_data.shape[0]/ self.__width) 
        
        '''
        math.floor: will round down to the nearest integer
     
        '''
        
        
        self.__saving_location_dict = saving_location_dict 
        


    def creating_sample(self):
        
    
        start = 0
    
        end = self.__width 
    
        '''
        end = width and not width -1 because the slcing in numpy is exclusive
        
        - so if width = 100 (we want 100 frames)
            [0:100] will be from 0 -->99 which are 100 frames
    
        '''

    
        for index in range(self.__iteration_per_csv_file):
        
        
            '''
            taking the unix tim stamp column
            '''
            
            unix_time_stamp_measured = self.__original_numpy_data[start:end,0].copy()
            
            '''
            Computing the difference 
            '''
            diff = np.diff(unix_time_stamp_measured)
            
            
            '''
            Checking for frame continuity
            '''
            
            
            test_shape = diff[diff > self.__margin].shape
    
    
            if test_shape[0] == 0:
                
        
                '''
                -All the frames are continous
                    -No violation 
            
                Here we create the file 
                '''
                filename = os.path.join(self.__saving_location_dict ['Directory'],
                                        self.__saving_location_dict ['File_Name'])
                
                                                             
                                                             
                np.save(filename +  str(index), \
                        self.__original_numpy_data[start:end, 3:].T)
        
                print('- Iteration # ',index,
                'Frames are continous \n')
        
            else:
                
                '''
                - There is discontinuity
                    Reject the frame                
                '''            
                
                print('- Iteration #',index,
                'Theres is discontinuity, slice is rejected \n')    
                
                
        
            
            # updating the offset         
            start = start + self.__width
            
            end =  end + self.__width
    
    
    
    
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





       
       
       