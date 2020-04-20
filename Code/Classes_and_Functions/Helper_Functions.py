





import numpy as np

import os

def creating_sample(original_numpy_data,width,iteration_per_csv_file,margin
,saved_path):
    
    start = 0
    
    end = width 
    
    '''
    end = width and not width -1 because the slcing in numpy is exclusive
        
        - so if width = 100 (we want 100 frames)
            [0:100] will be from 0 -->99 which are 100 frames
    
    '''

    
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
        
        
        test_shape = diff[diff > margin].shape


        if test_shape[0] == 0:
            
    
            '''
            -All the frames are continous
                -No violation 
        
            Here we create the file 
            '''
            filename = os.path.join('Created_Dataset','train_spec_')
            
            np.save(filename +  str(index), \
                    original_numpy_data[start:end, 3:].T)
    
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
        start = start + width
        
        end =  end + width
    
    
    
    
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





       
       
       