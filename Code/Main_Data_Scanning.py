




import os 

import collections

import itertools



'''
list containing sensor names
'''
list_sensor_name = ['urn:osh:sensor:noisemonitoring:B8-27-EB-EA-EB-EA',
                  'urn:osh:sensor:noisemonitoring:B8-27-EB-EA-12-88']


'''
Creating an ordered dictonary for the necessary paraemeters
in which will change during testing file existence

    - It is ordered since we want to preserve the order of data
    while we are doing iterations and constructing the path
'''

od2 = collections.OrderedDict()


od2['sensor_index'] = [0,1]

od2['year'] = [2019]

od2['month'] = [12]

od2['days'] = [v for v in range(1,2)] # 1 --> 28

od2['hour'] = [v for v in range(4)] # 0 --> 23



'''
Constructing a list  using the values of the odrdered
dictoonary
''' 
hyperparameters_values = [v for v in od2.values()]


'''
Computing the cartesian product among all the parameters

This cartesion product will be the data path in which we do the testing
'''
hyperparameters_values_gene = itertools.product(*hyperparameters_values)



print('- printing the cartesion product: \n \n')

for sensor_index, year, month , days , hour in \
itertools.product(*hyperparameters_values):
    
    
    print('- Working with sensor #:',sensor_index,'|',  
              'year #',year,'|',
              'month #',month,'|',
              'day #',days,'|', 
              'hour #',hour,'| \n')
    
    
    data_path = 'Data_Set/' + list_sensor_name [sensor_index] + '/' + \
      str(year) + '/' + str(month) + '/' + \
          str(days) + '/' + str(hour) + '.zip' 
          
    if os.path.isfile(data_path) == True:
        
        print('File Exist \n')
        
        
    else:
        
        print('No File \n')
    
    
   
  







