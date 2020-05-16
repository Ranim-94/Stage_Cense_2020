




import os

from Class_Organize_Files import Organize_Files


working_directory = 'raw'

list_all_files = os.listdir(working_directory)


print('--> We have ',len(list_all_files),'raw files avaialbale \n')

'''
Choose sepecified number of directories to be created
'''

director_param = {
    
    'option_directory_gen': False,
    
    'start': 13,
    
    'end':17

    }

nb_files_to_be_moved = 10


'''
Instantiate
'''

organize_intance = Organize_Files(director_param,working_directory,
                 nb_files_to_be_moved)



organize_intance.create_directories()
          
        
organize_intance.move_raw_files_log()

      
        

    
    

        



