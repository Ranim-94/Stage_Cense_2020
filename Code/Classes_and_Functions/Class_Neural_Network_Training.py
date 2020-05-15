
import torch

import math

from Classes_and_Functions.Class_Custome_Pytorch_Dataset import \
Dataset_SpecSense

from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Class_RunManager import RunManager

from Classes_and_Functions.Class_RunBuilder import RunBuilder





class Neural_Network_Training:
       
       
       def __init__(self,optimization_option,parameters_neural_network,
                    saving_location_dict,params_to_try,show_trace):
              
              '''
              this is the neural network model implemented in its
              proper class
              
              '''
              
              self.parameters_neural_network = parameters_neural_network
              
              self.optimization_option = optimization_option
              
              self.params_to_try = params_to_try
              
              self.saving_location_dict = saving_location_dict

              self.show_trace = show_trace
              
       
       def training(self):
           
              # objective_function = self.optimization_option['Objective_Function']
              
              '''
              Instantinating the RunManager class
              '''
              manager_object = RunManager()

              for run in RunBuilder.get_runs(self.params_to_try):
                               
                  if self.show_trace == True:
                      
                      print('--> Testing Combination of:',run,'\n')
                      
                  
                   # this will give us an iterable object
                  train_loader = \
                  torch.utils.data.DataLoader(dataset = 
                  Dataset_SpecSense(self.saving_location_dict), 
                  
                  batch_size = run.batch_size,shuffle = run.shuffle) 
                  
                  '''
                  Adding a key named batch size:
                      - this is due to 2 resons:
                             1) we are testing training for several batch size
                             2) Audio2Vec implementation in the forward() 
                                depends on the batch size
                             3) so for each batch size choice, we have a different
                                model to test
                  '''
                  
                  self.parameters_neural_network['batch_size'] = run.batch_size
                  
                  
                  # Creating the Neural Network instance
                  net_1 = Audio2Vec(self.parameters_neural_network)
                      
                  
                  '''
                    Adapting the optimizer to each neural network instance
                    created by adding some extra keys value to the optimization
                    option
                  '''
                  
                  self.optimization_option['optimizer'] =  \
                  torch.optim.Adam(net_1.parameters(), lr = math.pow(10,-3))
                    

                  
                  manager_object.begin_run(run, network = net_1,
                                           loader = train_loader)
                 
              
                  '''
                  Start training loop
                  '''
                  
                  self.__start_epoch(run,train_loader,manager_object,
                                     net_1)
                 
  
                  # End Block for epoch in range
                  manager_object.end_run()
              
              
              
                            
             # End Block for run in RunBuilder.get_runs(params) 
              manager_object.save('Results')
              
              print('------------------------------------ \n')
              
              print('-->  Done Training ! \n')
              
              print('------------------------------------ \n')
              
              
       def __start_epoch(self,run,train_loader,manager_object,net_1):
           
           
            for epoch in range(run.epoch_times):
                      
                         manager_object.begin_epoch()
                         
                  
                   # Variable to track the losses and the prediction in the Network
                         total_loss = 0
                  
                         for count,batch in enumerate(train_loader) :
                             
                             
                                if self.show_trace == True:
                                    
                                    print('--> batch #',count,'\n') 
                                
                                # unpacking
                                sample , labels = batch 
                                
                                if sample.shape[0] != run.batch_size:
                                    break
                                
                                if self.show_trace == True:
                                
                                    print('-->Sample shape is:',sample.shape,
                                      '| Labels shape is:',labels.shape,'\n')
                         
                         
                                # Pass a batch , do forward propagation
                                preds , _ = net_1(sample)
                                
                                
                                if self.show_trace == True:
                                
                                    print('--> Prediction shape is:',preds.shape,'\n')
                         
    
                         
                                 # Compute the loss
                                loss = \
                                self.optimization_option['Objective_Function'](preds,labels)
                                
                                if self.show_trace == True:
                                
                                    print('--> Loss is:',loss,'\n')
                  
                                self.optimization_option['optimizer'].zero_grad()
                         
                                '''
                                 We use the zero_grad() method because in PyTorch
                  
                                each step we are computing the grad of the weigths
                  
                                PyTorch will accumulate the current grad value with the 
                  
                                previous value of the grad, and we don't  want this
                  
                                we need only the current value not the accumalation
                  
                                so we zero the gradient before each weight update
                                
                                '''
                         
                                # Computing the gradient
                         
                                loss.backward()
                         
                                # Update the weights using the optmizer
                         
                                self.optimization_option['optimizer'].step()
                  
                                total_loss += loss.item()
                  
          
                                print('epoch: ',epoch,'| total_Loss :',total_loss,'\n')
                                
                                manager_object.end_epoch()
           
           
           
        
        
                            
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              