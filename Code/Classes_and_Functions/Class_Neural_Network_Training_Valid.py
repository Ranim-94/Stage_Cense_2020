



import torch

import math

import os

from Classes_and_Functions.Class_Custome_Pytorch_Dataset import \
Dataset_SpecSense

from Classes_and_Functions.Class_Audio2Vec import Audio2Vec

from Classes_and_Functions.Class_RunManager import RunManager

from Classes_and_Functions.Class_RunBuilder import RunBuilder

from Classes_and_Functions.Helper_Functions import get_all_preds

from Classes_and_Functions.Helper_Functions import get_num_correct

from Classes_and_Functions.Class_Sensor_Classification import My_Calssifier_Encoder


class Neural_Network_Training_Valid:
       
       
       def __init__(self,optimization_option,parameters_neural_network,
                    saving_location_dict,params_to_try,
                    frame_width ,rows_npy,show_trace,model_names,
                    save_point,start_from_iter,resume_training,
                    loaded_model,mode):
              
              '''
              this is the neural network model implemented in its
              proper class
              
              '''
              
              self.parameters_neural_network = parameters_neural_network
              
              self.optimization_option = optimization_option
              
              self.params_to_try = params_to_try


              self.saving_location_dict = saving_location_dict

              self.show_trace = show_trace

              self.mode = mode
              '''
              To choose whether we do sensor classification
              or pretext task
              '''
              
              # In case we need to continue training from a certain checkpoint
              self.resume_training = resume_training
              
              # The model we trained before
              self.loaded_model = loaded_model
              
              self.model_names = model_names
              
              self.save_point = save_point
              
              self.start_from_iter = start_from_iter
              
              # For dataset to consruct the data
              self.frame_width , self.rows_npy = frame_width , rows_npy
        
        
       def training(self):
           
              
              '''
              Instantinating the RunManager class
              '''
              manager_object = RunManager()
              
              
              '''
              Setting the GPU name or CPU 
              '''
              
              if torch.cuda.is_available() == True:

                  device = torch.device("cuda:0")
                  
                  print(f'Training will be on GPU \n')
                  
              else:
                 
                 device = torch.device("cpu")
                 
               
              '''
              Sepcify which task we are using this class
              so we can define the correspondant objective function
              '''  
              if self.mode == 'pretext_task':
                      
                      objective_function = \
                      self.optimization_option['Objective_Function_reconstruction']
                      
              elif self.mode == 'sensor_classification':
                      
                      objective_function = \
                      self.optimization_option['Objective_Function_sensor_classification'] 
                
              
              '''
              Start testing for different combination of parameters
              '''      
              for count_run,run in enumerate(RunBuilder.get_runs(self.params_to_try)) :

                  
                  print(f'--> # Run {count_run} | Testing Combination of: {run} \n')
                      
                  
                  '''
                  Setting the correspondant directory for the 
                  specific data perecentage we are using for training
                  '''
                  
                  self.saving_location_dict['Directory'] = \
                     f'CreatedDataset/Training_Set_{run.data_percentage}'
                    
                    
                   # this will give us an iterable object
                  self.train_loader = \
                  torch.utils.data.DataLoader(dataset = 
                  Dataset_SpecSense(self.saving_location_dict,
                                    self.rows_npy ,self.frame_width,self.mode), 
                  
                  batch_size = run.batch_size,shuffle = run.shuffle) 
                  
                  
                  
                  '''
                  Create Valid loader to compute accracy and
                  losses on Validation set
                  after each epoch
                  '''

                  self.saving_location_dict['Directory'] = \
                     f'CreatedDataset/Validation_Set'
                    
                    
                   # this will give us an iterable object
                  self.valid_loader = \
                  torch.utils.data.DataLoader(dataset = 
                  Dataset_SpecSense(self.saving_location_dict,
                                    self.rows_npy ,self.frame_width,self.mode), 
                  
                  batch_size = run.batch_size,shuffle = run.shuffle) 
                  
                  # Computing epoch_nb 
                  self.epoch_nb = len(self.train_loader)// run.batch_size
                      
                  
                  if self.mode == 'pretext_task':
                      
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
                      
                      # Creating the Neural Network instance for pretext task
                      net_1 = Audio2Vec(self.parameters_neural_network)
                      
                      name = self.model_names['embedding']
                      
                      
                      # In case we need to resume training
                      if self.resume_training == True:
                          
                          # Lading the checkpoint
                          checkpoint = \
                          torch.load(self.loaded_model)
                          
                          net_1.load_state_dict(checkpoint['model_state'])
                          
                          # Setting the nework in training mode
                          net_1.train()
                          
                          
                      
                      
                      
                  elif self.mode == 'sensor_classification':
                      
                      
                      self.parameters_neural_network['batch_size'] = run.batch_size

                      net_1 = My_Calssifier_Encoder(self.parameters_neural_network)
                      
                      name = self.model_names['classification_no_embedding']
                      

               
                  
                  # Moving the model (the model weights) to the GPU
                  net_1.to(device)
                      
                  
                  '''
                    Adapting the optimizer to each neural network instance
                    created by adding some extra keys value to the optimization
                    option
                  '''
                  self.optimization_option['optimizer'] =  \
                      torch.optim.Adam(net_1.parameters(), lr = math.pow(10,-3),
                                       amsgrad = True)
                  
                  # In case we need to resume training
                  if self.resume_training == True:
                      
                        self.optimization_option['optimizer'].load_state_dict(checkpoint['optim_state'])
                      
      
                  manager_object.begin_run(run, network = net_1,
                                           loader = self.train_loader)
                 
        
                  # Start training loop
                  valid_loss_per_epoch, accuracy_validation = \
                  self.__start_train_valid(count_run,run,manager_object,
                                     net_1,device,objective_function,name)
                  
                  
     
              # End of all runs for the training
              print('********************************* \n')
              
              print('-->  Done Training ! \n')
              
              print('********************************* \n')
              
              
              
              return valid_loss_per_epoch,accuracy_validation
              
              
       def __start_train_valid(self,count_run,run,manager_object,net_1,
                         device,objective_function,name):
           
           
            nb_of_iter , actual_iter = run.nb_of_iter - 1 , self.start_from_iter
            
            
            '''
            nb_of_iter : required nb of iteration
                - I have put -1 because I start counting from 0
            
            In Audio2vec paper: nb_of_iter = 3 million 
            '''
            
            accuracy_validation = []
            
            # Very high initial loss for Audio2Vec model
            # And also accuracy for sensor classification
            loss_start, best_acc = 10**4 , 0
            
            # to track losses
            valid_loss, valid_loss_per_epoch = [], []
            
            # I just initilize as initial value
            # so it won't give error in the scope of
            # validation and sensor classification task
            df_iter = []
                        
            while actual_iter < nb_of_iter:
                
                for phase in ['train','valid']:
                    
                    
                    if phase == 'train':
                        
                        # Setting the model in training mode
                        net_1.train()
                        
                        loader = self.train_loader
                        
                        print(f'We are in training mode | Nb of Sample_train:',
                          f'{len(loader)} \n')
                        
                        
                    else:
                        
                        # Setting the model in eval mode
                        net_1.eval()
                        
                        loader = self.valid_loader
                        
                        print(f'We are in valid mode | Nb of Sample_valid:',
                          f'{len(loader)} \n')
                        
                    if phase == 'valid' and self.mode == 'sensor_classification':
                        
                        '''
                        In this block, I compute the accuracy on 
                        validation set for sensor classification 
                        
                        and save results when
                        accuracy increase
                        '''
            
                        # Computing the prediction for all batches
                        # using the trained model
                        test_pred, test_labels = \
                            get_all_preds(net_1,self.valid_loader,device)
                            
                            
                        print(f'test_pred shape: {test_pred.shape} | ',
                              f'test_labels shape: {test_labels.shape}')
                                            
                        preds_correct_test = \
                            get_num_correct(test_pred,test_labels)
                                                               
                        acc_nb = preds_correct_test/(run.batch_size*len(self.valid_loader))
                        
                        # Saving the model and some results
                        # when accuracy increased
                        if acc_nb > best_acc:
                            
                            best_acc = acc_nb
                            
                            checkpoint = {
                                                
                                'iter': actual_iter,
                                                
                                'model_state':net_1.state_dict(),
                                                
                                'perecentage':run.data_percentage, 
                                'pandas':df_iter,'name':name,
                                            
                                'optim_state':
                                self.optimization_option['optimizer'].state_dict()
                                                
                                            }
                                                
                            torch.save(checkpoint, 
                                 f'Saved_Iteration/{name}_{run.data_percentage}.pth')
                            
                        # Appending the result
                        accuracy_validation.append(acc_nb)
                                        
                        print(f'Accuracy on validation set is: {acc_nb} \n')
                    
 
                    
                    else:

                        # This loop will be entered in the following 
                        # cases:
                                        
                        #  1) training and pretext task
                        #  2) training and sensor classification task
                        #  3) validation on pretext task
                        #     to compute losses (MSE) on 
                        #     validation set in each iteration
                        #     then taking the average for each epoch

                         for count,batch in enumerate(loader):
     
                                 
                                    if self.show_trace == True:
                                        
                                        print(f'--> batch # {count} \n') 
   
                                    # unpacking
                                    sample , labels = batch 
                                    
                                    # print(f' --> labels shape: {labels.shape} \n')
                                    
                                    # print(f'--> Labels are: {labels} \n')
      
                                    # Moving the Data to GPU
                                    sample , labels = sample.to(device), \
                                    labels.to(device)
    
                                    
                                    # We count iteration only in training mode
                                    if phase == 'train':
                                        
                                    
                                        '''
                                        If we reach required nb of iteration
                                        we break from the internal for loop
                                        which process the batches
                                        '''
                                        if actual_iter > nb_of_iter or \
                                        sample.shape[0] != run.batch_size:
                                            break

                                    
                                    if self.show_trace == True:
                                    
                                        print(f'-->Sample shape is: {sample.shape}',
                                          f'| Labels shape is: {labels.shape} \n')
                             
                                    
                                    if phase == 'valid':
                                        
                                        if sample.shape[0] != run.batch_size:
                                            break
                                    
                                    # Pass a batch , do forward propagation
                                    preds  = net_1(sample)
                                    
                                    
                                    if self.show_trace == True:
                                    
                                        print(f'--> preds shape: {preds.shape} \n')
                                    
                                    
                                    
                                    if phase == 'train':
                                        # here we count +1 iteration
                                        actual_iter += 1
                                    
                                        '''
                                        Also I count here because the number of iterations
                                        I will put in the pandas data frame bulided in 
                                        the RunManager Class
                                        '''
                                        manager_object.begin_iter()
                                        
                                        if self.mode == 'sensor_classification':
                                        
                                            print(f'# Train Sensor Classif | Iteration # {actual_iter}' ,
                                                  f' | run # {count_run}\n')
                                            
                                            
                                        else:
                                            
                                            print(f'# Train Pretext Task | Iteration # {actual_iter}' ,
                                                  f' | run # {count_run}\n')
                                        
                                        
                                    if phase == 'valid':
                                                                                
                                        print(f'--> Valid | Iteration: {count} \n')
                                    

                                    if self.mode == 'sensor_classification':
                                        
                                        # Compute the loss
                                        loss = \
                                        objective_function(preds,labels.long())
         
            
                                    elif self.mode == 'pretext_task':
                                        
                                        # Compute the loss
                                        loss = \
                                        objective_function(preds,labels)
                                             
                                        
                                        

                                    if self.show_trace == True:
                                    
                                        print('--> Loss is:',loss,'\n')
                                        
                                        
                                    if phase == 'train':    
                      
                                        self.optimization_option['optimizer'].zero_grad()
                                 
                                        '''
                                         We use the zero_grad() method because in PyTorch
                          
                                        each step we are computing the grad of the weigths
                          
                                        PyTorch will accumulate the current grad value with the 
                          
                                        previous value of the grad, and we don't  want this
                          
                                        we need only the current value not the accumalation
                          
                                        so we zero the gradient before each weight update
                                        
                                        '''
                                 
                                        # Computing the gradient of the loss with
                                        # % to network weights
                                 
                                        loss.backward()
                                 
                                        # Update the weights using the optmizer
                                 
                                        self.optimization_option['optimizer'].step()
                                        
                                        
                                        '''
                                        1) Recompute the loss after upadting the weights
                                        
                                        2) extract it as a python float number using the
                                        
                                        item() method
                                        
                                        3) Also I save this result in the results
                                            dictonary created inisde Class_RunManager 
                                        
                                        '''
                                        
                                        # print(f'--> Loss.item is: {loss.item()} \n')
                                        
                                   
                                        
                                        df_iter = manager_object.end_iter(loss.item())
    
                                  
                                    if phase == 'valid':
                                        
                                        if self.mode == 'pretext_task':
                                            
                                            valid_loss.append(loss.item())
       
  
                            
    
                     # '''
                     #    In this scope the epoch is finished
                     #    We compute here the average loss per 
                     #    epoch on the validation set for pretext task
                     # '''                
                    if phase == 'valid':
                            
                            if self.mode == 'pretext_task':
                                
                                # computing average valid loss per epoch
                                av_loss = sum(valid_loss)/len(valid_loss)
                                     
                                if av_loss < loss_start:
                                    
                                    loss_start = av_loss
                                    
                                    # Saving  only when loss is decreasing
                                    checkpoint = {
                                                
                                            'iter': actual_iter,
                                                
                                            'model_state':net_1.state_dict(),
                                                
                                            'perecentage':run.data_percentage, 
                                            'pandas':df_iter,'name':name,
                                            
                                            'optim_state':
                                            self.optimization_option['optimizer'].state_dict()
                                                
                                                }
                                                
                                    torch.save(checkpoint, 
                                 f'Saved_Iteration/{name}_{run.data_percentage}.pth')
                                    
                                    
                                
                                # Appending the result for this epoch
                                valid_loss_per_epoch.append(av_loss)
                                         
                                # Clearning the valid_loss list
                                valid_loss = []
                                         
                                print('--> Done with Valid Mode \n')

            
            
            
            
            
            if phase == 'train':

                # Particular Combination has finished 
                manager_object.end_run()
            
            
            return valid_loss_per_epoch,accuracy_validation
            
            
           
           
        
        
                            
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
             