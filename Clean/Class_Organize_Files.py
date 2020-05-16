

import os

import shutil

class Organize_Files:
    
    
    def __init__(self,director_param,working_directory,
                 nb_files_to_be_moved):
        
        
        self.list_all_files = os.listdir(working_directory)
        
        self.director_param = director_param
        
        
        self.working_directory = working_directory
        
        self.nb_files_to_be_moved = nb_files_to_be_moved
        
        
        
        
        '''
        Creatiing an array of working set names
        '''
        
        Working_set =  []
        
        index_directories = tuple([nb for nb in 
                 range(director_param['start'],director_param['end'] + 1)]) 
        
        
        
        for item in index_directories:
            
            Working_set.append('Set_' + str(item))
            
            
        self.Working_set = Working_set   





    def create_directories(self):
        
        '''
        Generate folders having the same name
        as in Working_set list
        '''
        
    
        if self.director_param ['option_directory_gen'] == True:
            
            for item in self.Working_set:
            
                os.mkdir(item) 
            
            
    def move_raw_files_log(self):
        
        '''
        Moves a sepefic number of files to each
        directory created and generate a log file
        '''
        
        
        outfile = open("Sounds.txt", "w")
        
        for item in self.Working_set:
            
            print('--> We are in ' + item,'\n')
            
            for i in range(self.nb_files_to_be_moved):
                
                source_raw_file = self.working_directory + '/' + \
                self.list_all_files[i]
                
                shutil.move(source_raw_file,item)
                
                del self.list_all_files[i]
                

                
                
                
                
            outfile.write('---- Start' + item +'---- \n \n')
            
            
            '''
            - Listing the .wav files in each directory
                
            - Splitting the .wav extension from the files names
                
            - saving the name without .wav extension
            '''

            wav_files_names = [os.path.splitext(item)[0] for item \
                  in os.listdir(os.getcwd() + '/' + item)]
                
    
    
            '''
            Generating a log file 
            '''    
                    
            for item in wav_files_names:
                    
                outfile.write(item +'--> \n \n')
                    
                    
            outfile.write('\n \n')
                    
        
        
        
        outfile.close() 
 
            
 
    

        

                
                
                
                
                
                    
                
                     
         