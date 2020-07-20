



from Classes_and_Functions.Helper_Functions import plot_results


import torch

data_percent = 6


checkpoint_model = torch.load(f"Saved_Iteration/classif_no_emb_50_6.pth")

frame_result = checkpoint_model['pandas']

param_plot = {
    
    
    'name_file': f'classif_no_emb_100_6.pth',
    
    'title':f'Task: Reconstruction loss for {data_percent} \% of data',
    
    'xlabel':'Iteration Number',
    
    'ylabel':'Cross Entropy',
    
    'save':f'Classif_{data_percent}_percent.pdf'
    
    
    }


plot_results(param_plot)









