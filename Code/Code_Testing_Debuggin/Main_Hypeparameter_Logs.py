

'''
product funtion will give us the cartesian product

Helpful when doing a list of combination

'''
from itertools import product

from collections import OrderedDict

import math



from Classes_and_Functions.Class_RunBuilder import RunBuilder


# Creaing a dictionary of hyperparameters to test

params_to_try = OrderedDict(
    
    batch_size = [20],
    
    
    # percentage of data we need to test
    
    # data_percentage = [6,12,50,100],
    
    data_percentage = [12,25],

    # rquired nb of iteration ,
    # it is independent of batch size or nb of epoch
    nb_of_iter = [ 2 * int(math.pow(10,1)) ], 
    

    shuffle = [False]
)



# Using the RunBuilder class

runs = RunBuilder.get_runs(params_to_try)


print(f'--> {runs} \n')

for count,run in enumerate(RunBuilder.get_runs(params_to_try)) :
       
       print(f'--> # {count}: {run} \n')
       
       print(f'{run.data_percentage} \n')









