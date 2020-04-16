




import torch



def creating_sample(original_numpy_data,width):
    
    tensor_column_sample = torch.from_numpy(original_numpy_data[:width, 3:].T).reshape(-1,1)
    
    
    '''
    - Description of the function:    
    
    -At a first step we matrix of shape 29 x width
        - where width is the number of lines in the csv files
        - it will be specified
        
        - the 3 index in [[:self.width, 3:]] 
        is where the octave bands starts in the csv file
            - we have 29 octave bands we are working on
                that's why the shape will be 29 x width
                
            - I have reshaped this sample to a column vector
                - shape: (29 x width) , 1
    '''

    return tensor_column_sample



       
       
       