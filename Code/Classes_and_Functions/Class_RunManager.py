

import time

from IPython.display import display, clear_output

import pandas as pd




from collections import OrderedDict

class RunManager():
       
       # Constructor: seeting the attirbutes
       def __init__(self):
              
              
              '''
              Tracking epoch life cycle
              
              We will do this for every iteration
              ''' 
              self.iter_count = 0
              
              self.loss = 0
      
              
              
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
 
              
              
       def begin_run(self, run, network, loader):
              
           
              
              self.run_params = run
              
              self.run_count += 1
              
              self.network = network
              
              self.loader = loader
              
         
           
       def end_run(self):
           
              self.iter_count = 0
              
              
       def begin_iter(self):
              
              self.epoch_start_time = time.time()
              
              self.iter_count += 1

              
              # self.epoch_num_correct = 0
              
       def end_iter(self,loss):
              

              
              results = OrderedDict()
              
              results["run"] = self.run_count
              
              results["Iteration"] = self.iter_count
              
              results['loss'] = loss

              for k,v in self.run_params._asdict().items(): results[k] = v
              
              self.run_data.append(results)
              
              df_iter = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
              
              return df_iter
             
              #  # For Ipython console
              # clear_output(wait=True) # clear the current output
              
              # display(df) # display the new data frame 
              
              
       # def track_loss(self, loss):
              
       #        '''
       #        This version is the loss relative to the batch size                                
       #        '''             
       #        # self.epoch_loss += loss.item() * self.loader.batch_size
              
       #        #self.epoch_loss += loss.item()
              
       #        '''
       #        Trying non-cumulative loss
       #        '''
              
       #        '''
       #        Recompute the loss after upadting the weights
                                    
       #        and extract it as a python float number using the
                                    
       #        item method
       #         '''
              
       #        self.loss = loss.item()

     
       
       
       def save(self, fileName):
              
              '''
              Saving the results in csv format
              '''
              
              
              pd.DataFrame.from_dict(self.run_data, 
                                     orient='columns').to_csv('Results_training.csv')

             
                      
        
              
              
       
              
              
              
              
              
              
              
              
              
              