




import math


from Classes_and_Functions.Class_Dataset_Construction import SpecCense_Construction



'''
Constructing a dictionary of the data path
containing neccessary parameters
'''

path_dictionary = {


    'sensor_index':0,
    
    'sensor_name' : ['urn:osh:sensor:noisemonitoring:B8-27-EB-EA-EB-EA',
                    'urn:osh:sensor:noisemonitoring:B8-27-EB-EA-12-88'],
    
     # this is fixed for now for all sensors
    'year' : str(2019), # this is fixed
    
    'month_option' : str(12) , # this is fixed,

    'day_option' : str(24) ,

    'hour_option' : str(3) ,
        
}




data_path = 'Data_Set/' + path_dictionary['sensor_name'][path_dictionary ['sensor_index']] + '/' + \
path_dictionary['year'] + '/' + path_dictionary['month_option'] + '/' + \
path_dictionary ['day_option'] + '/' + path_dictionary['hour_option'] + '.zip'




width , margin  = int(math.pow(10,4)) , 250

'''
Casting into int because I use width as index in slicing
'''



saving_location_dict = {
    
    'Directory': 'Created_Dataset',
    
    'File_Name':'train_spec_'
    
    
    }


# Creating the instance
data_instance = SpecCense_Construction(data_path = data_path , width = width , \
                  margin = margin, saving_location_dict = saving_location_dict  )
    

# Calling the method

data_instance.creating_sample()










