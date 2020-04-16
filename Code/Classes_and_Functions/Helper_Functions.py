




import torch

import numpy as np

def creating_sample(original_numpy_data,width,iteration_per_csv_file):
    
    start = 0
    
    end = width 
    
    '''
    end = width and not width -1 because the slcing in numpy is exclusive
        
        - so if width = 100 (we want 100 frames)
            [0:100] will be from 0 -->99 which are 100 frames
    
    '''
    
   
    
    shape = (29 * width,iteration_per_csv_file)    
    
    tensor_matrix_sample = torch.from_numpy(np.empty(shape))
    
    

    
    for index in range(iteration_per_csv_file):
        
        tensor_matrix_sample[:,index] = \
        torch.from_numpy(original_numpy_data[start:end, 3:].T).reshape(-1,1).squeeze()
        
        '''
        .squeeze() will eliminate all axis except the last one
        
        '''
        
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
                
            - I have reshaped this sample to a column vector
                - shape: (29 x width) , 1
    '''

    return tensor_matrix_sample



       
       
       