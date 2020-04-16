


import torch


from Classes_and_Functions.Helper_Functions import creating_sample 


class SpecCense_Dataset(torch.utils.data.Dataset):
    
    
    '''
    This class is the responsible for data generation from 
    the SpecCense data

    - Recall that in pytorch we inhere from 
    the Dataset class and we reimplement the methods

    so we can then use Dataloader in later phase    
    
    '''

    def __init__(self,original_numpy_data,width):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
 
      
        

        
        self.x_data = creating_sample(original_numpy_data,width)
        
        
        
        '''
        -At a first step we matrix of shape 29 x width
            - where width is an argument to be specified
        
        - the 3 index in [[:self.width, 3:]] 
        is where the octave bands starts in the csv file
            - we have 29 octave bands we are working on
                that's why the shape will be 29 x width
                
            - I have reshaped this sample to a column vector
                - shape: (29 x width) , 1
        '''
        
 


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
        return self.x_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        '''
        this method will compute the number of samples we have
        '''   
        return self.n_samples
 
       
       
       
       