



from Classes_and_Functions.Helper_Functions import plot_results



data_percent = 12

param_plot = {
    
    
    'name_file': f'classif_no_emb_80000_{data_percent}.pth',
    
    'title':f'Task: Sensor Classification loss for {data_percent} \% of data',
    
    'xlabel':'Iteration Number',
    
    'ylabel':'Cross Entropy',
    
    'save':f'Sensor_Classif_no_emb_{data_percent}_percent.pdf'
    
    
    }


plot_results(param_plot)









