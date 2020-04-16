


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

    def __init__(self,original_numpy_data,width,iteration_per_csv_file):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        self.n_samples = iteration_per_csv_file
        
        self.x_data = creating_sample(original_numpy_data,width,iteration_per_csv_file)
        

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
        return self.x_data[:,index]

    # we can call len(dataset) to return the size
    def __len__(self):
        '''
        this method will compute the number of samples we have
        '''   
        return self.n_samples
 
       
       
       
       