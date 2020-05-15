

import time

from IPython.display import display, clear_output

import pandas as pd




from collections import OrderedDict

class RunManager():
       
       # Constructor: seeting the attirbutes
       def __init__(self):
              
              
              '''
              Tracking epoch life cycle
              
              We will do this for every epoch
              ''' 
              self.epoch_count = 0
              self.epoch_loss = 0
              self.epoch_num_correct = 0
              self.epoch_start_time = None
              
              
              
              self.run_params = None
              '''
              The value of this attribute will be one of 
              the values returned from RunBuilder Class
              '''
              
              self.run_count = 0
              
              self.run_data = []
 
              '''
              We will append for each run the value of the params
              we are testing
              
              '''
              self.run_start_time = None
              
              
              
              
              self.network = None
              self.loader = None
              self.tb = None
              
              
       def begin_run(self, run, network, loader):
              
              self.run_start_time = time.time()
              
              self.run_params = run
              self.run_count += 1
              
              self.network = network
              self.loader = loader
              
         
           
       def end_run(self):
              self.epoch_count = 0
              
              
       def begin_epoch(self):
              
              self.epoch_start_time = time.time()
              
              self.epoch_count += 1
              self.epoch_loss = 0
              
              # self.epoch_num_correct = 0
              
       def end_epoch(self):
              
              epoch_duration = time.time() - self.epoch_start_time
              
              run_duration = time.time() - self.run_start_time
              
              loss = self.epoch_loss / len(self.loader.dataset)

              
              results = OrderedDict()
              results["run"] = self.run_count
              results["epoch"] = self.epoch_count
              results['loss'] = loss
              results['epoch duration'] = epoch_duration
              results['run duration'] = run_duration
              
              for k,v in self.run_params._asdict().items(): results[k] = v
              
              self.run_data.append(results)
              
              df = pd.DataFrame.from_dict(self.run_data, orient='columns')
             
              # For Ipython console
              clear_output(wait=True) # clear the current output
              display(df) # display the new data frame 
              
              
       def track_loss(self, loss):
              
              self.epoch_loss += loss.item() * self.loader.batch_size

     
       
       
       def save(self, fileName):
              
              '''
              Saving the results in 2 format: csv and json file
              '''
              
              
              pd.DataFrame.from_dict(self.run_data, 
                                     orient='columns').to_csv('Results_training.csv')

             
                      
        
              
              
       
              
              
              
              
              
              
              
              
              
              