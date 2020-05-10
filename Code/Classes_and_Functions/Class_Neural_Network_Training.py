
import torch

from Classes_and_Functions.Class_Custome_Dataset import Dataset_SpecSense


class Neural_Network_Training:
       
       
       def __init__(self,optimization_option,saving_location_dict):
              
              '''
              this is the neural network model implemented in its
              proper class
              '''
              self.optimization_option = optimization_option
              
              self.saving_location_dict = saving_location_dict
              
       
       def training(self):
           

               # this will give us an iterable object
              train_loader = \
              torch.utils.data.DataLoader(dataset = 
                        Dataset_SpecSense(self.saving_location_dict), 
                          batch_size = self.optimization_option['batch_size'],
                          shuffle = True) 
              
              
              objective_function = self.optimization_option['Objective_Function']
              
              
              '''
              Start training loop
              '''
              
              for epoch in range(self.optimization_option['epoch_times']):
                     
              
               # Variable to track the losses and the prediction in the Network
                     total_loss= 0
              
                     for batch in train_loader:
                            
                            # unpacking
                            sample , labels = batch 
                            
                            print('--> Labels shape is:',labels.shape,'\n')
                     
                     
                            # Pass a batch , do forward propagation
                            preds , _ = \
                            self.optimization_option['neural_network_model'](sample)
                            
                            print('--> Prediction shape is:',preds.shape,'\n')
                     

                     
                             # Compute the loss
                            loss = objective_function(preds,labels)
                            
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
                            
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              