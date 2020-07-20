

import torch

class Other_Parameters:
    
    
    def __init__(self):
        
        
        self.saving_location_dict  = {
    
        # 'Directory': this key will be added in training process 
        # to use different data perecentage,
    
        'File_Name_Spectrograms':'train_spec_',
        
        'File_Name_time_stamp':'train_time_',
        
        'File_Name_sensor_id' : 'train_id_'
        }
        
        
        self.optimization_option = {

         # For Audio2Vec Embedding task   
       'Objective_Function_reconstruction':torch.nn.MSELoss(),
        
        
        # For Sensor Classification  task
        'Objective_Function_sensor_classification':torch.nn.CrossEntropyLoss()
        
        }
        
        
        self.model_names = {
    
        'classification_no_embedding':'classif_no_emb',
        
        'classification_with_embedding': 'classif_yes_emb',
        
        'embedding':'Audio2Vec_emb'

        }
        
        self.frame_width , self.rows_npy = 40, 10**4
        
        # If I want to display some print() statements for debugging purpose
        self.show_trace = False

            
        
        
        
        
