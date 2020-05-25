
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
           
              
              '''
              Instantinating the RunManager class
              '''
              manager_object = RunManager()
              
              
              '''
              Setting the GPU name
              '''
              
              GPU_device = torch.device("cuda:0")
              

              for run in RunBuilder.get_runs(self.params_to_try):
                               
                  if self.show_trace == True:
                      
                      print('--> Testing Combination of:',run,'\n')
                      
                  
                   # this will give us an iterable object
                  train_loader = \
                  torch.utils.data.DataLoader(dataset = 
                  Dataset_SpecSense(self.saving_location_dict,run.data_percentage), 
                  
                  batch_size = run.batch_size,shuffle = run.shuffle) 
                  
                  
                  
                  print('--> len(train_loader):',len(train_loader),'\n')
                  
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
                  Moving the model (the model weights) to the GPU
                  '''
                  net_1.to(GPU_device)
                      
                  
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
                                     net_1,GPU_device)
                 
  
                  # End Block for epoch in range
                  # manager_object.end_run()
              
              
              
                            
             # End Block for run in RunBuilder.get_runs(params) 
              manager_object.save('Results')
              
              print('********************************* \n')
              
              print('-->  Done Training ! \n')
              
              print('********************************* \n')
              
              
       def __start_epoch(self,run,train_loader,manager_object,net_1,
                         GPU_device):
           
           
            nb_of_iter , actual_iter = run.nb_of_iter , 0
            
            '''
            nb_of_iter : required nb of iteration
            
            In Audio2vec paper: nb_of_iter = 3 million 
            '''
            
            while actual_iter < nb_of_iter:
                

                for epoch in range(run.nb_of_epoch):
                          
                             manager_object.begin_epoch()
                             
    
                             for count,batch in enumerate(train_loader):
                                 
  
                                    if self.show_trace == True:
                                        
                                        print('--> batch #',count,'\n') 
                                    
                                    # unpacking
                                    sample , labels = batch 
                                    
                                    '''
                                    Moving the Data to GPU
                                    '''
                                    
                                    sample , labels = sample.to(GPU_device), \
                                    labels.to(GPU_device)
                                    
                                    
                                    
                                    '''
                                    If we reach required nb of iteration
                                    we break from the internal for loop
                                    which process the batches
                                    '''
                                    if actual_iter > nb_of_iter:
                                        break
                                    
                                    if sample.shape[0] != run.batch_size:
                                        break
                                    
                                    if self.show_trace == True:
                                    
                                        print('-->Sample shape is:',sample.shape,
                                          '| Labels shape is:',labels.shape,'\n')
                             
                             
                                    # Pass a batch , do forward propagation
                                    preds , _ = net_1(sample)
                                    
                                    actual_iter += 1
                                    
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
                                    
                                    manager_object.track_loss(loss)
                      
                                    '''
                                    End 1 epoch
                                    '''
                
                                    manager_object.end_epoch()
                 
                                    
                 
                if actual_iter > nb_of_iter:
                    break
                 
                '''
                Also we need to break from the outer loop 
                which deals with the epoch (the whole dataset)
                '''    

                   
                '''
                End All Epoch
                '''                    
                manager_object.end_run()
           
           
           
        
        
                            
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              