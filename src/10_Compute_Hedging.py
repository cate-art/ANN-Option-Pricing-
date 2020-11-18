import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from sklearn.utils import resample
import tensorflow as tf

from src.MLP import build, loadData, splitDataXY, splitTrainAndTest, CheckAccuracy, plotTrainingMetrics, getDer,getBondValueANN, getDerivativeFromSampleBS, getBondValueBS

# Load the dataset
data = loadData('../data/call.csv') 
description = data.describe()

# Get Training, Evaluation and Test splits
data = data[['optionid',
            'date',
            'strike',
            'close',
            'maturity',
            'interest',
            'BSgarch',
            'normmat',
            'close_norm',
            'strike_norm',
            'maturity_anunual',
            'volatility5',
            'volatility30',
            'volatility60',
            'volatility90',
            'volatility120',
            'vol_garch',
            'price_norm']]

# Add a column with the index for each sample. This is used for selecting the test and training test
data['index'] = range(1, len(data) + 1)

num_of_sample = int(0.75 * len(data))
num_of_epochs = 5
num_of_bootstraping_iter = 500
test_loss_min = 1

# Build the model
[X,Y] = splitDataXY(data)

 
# Build the model
model = build(X)
base_filename = 'bootstrap_model'
model.load_weights('../models/' + base_filename + '_weights.h5')
grad_func = tf.gradients(model.output[:, 0], model.input)

# Loop through all the options
import os.path 

# Check if the .csv file already exists
if os.path.isfile('../data/performance_data.csv'):
     performance_data = pd.read_csv('../data/performance_data.csv')
     print('INFO: csv file found.')
else:
     performance_data = pd.DataFrame(columns=['num_of_options', 'epsilon_BS', 'nu_BS', 'epsilon_ANN', 'nu_ANN', 'tracking_error_ANN', 'tracking_error_BS'])
     performance_data.to_csv('../data/performance_data.csv') 
     print('INFO: csv file not found. Initializing dataframe')


# Check if the .csv file already exists
if os.path.isfile('../data/performance_data_it.csv'):
     performance_data_it = pd.read_csv('../data/performance_data_it.csv')
     print('INFO: csv file found.')
else:
     performance_data_it = pd.DataFrame(columns=['optionid', 'epsilon_BS', 'nu_BS', 'epsilon_ANN', 'nu_ANN', 'moneyness', 'interest', 'maturity', 'tracking_error_ANN', 'tracking_error_BS'])
     performance_data_it.to_csv('../data/performance_data_it.csv') 
     print('INFO: csv file not found. Initializing dataframe')

for bootstraping_iter in range(num_of_bootstraping_iter):
    
    #performance_data_it = pd.DataFrame(columns=['optionid', 'moneyness', 'interest', 'maturity', 'tracking_error_ANN', 'tracking_error_BS'])
       
    boot_indexes = resample(data['index'], replace=True, n_samples=num_of_sample, random_state=bootstraping_iter)
    oob_indexes  = [x for x in data['index'] if x not in boot_indexes]
    
    boot = resample(data, replace=True, n_samples=num_of_sample, random_state=bootstraping_iter)
    test_hedge =  data.loc[data['index'].isin(oob_indexes)]
    
    [X_test, Y_test]   = splitDataXY(test_hedge)
    
    # Y_hat_NN = model.predict(Y_test)
    # test_hedge['price_NN_norm'] = Y_hat_NN[:,0]
    
    #REMOVE NON REPRESENTATIVE OBSERVATIONS
    # Take the price predicted in the NN
    #test_hedge['price_BS_norm'] = test_hedge['BSgarch']/test_hedge['strike']
    
    # Group by option id and conunt how many different options there are
    numb_options = test_hedge.groupby(['optionid'])['date'].count()
    #change name from date to count
    numb_options = pd.DataFrame(numb_options)
    numb_options = numb_options.rename(columns = {"date": "count"})
    #merge numb_options with test_hedge
    hedge_data = test_hedge.merge(numb_options,left_on='optionid', right_on='optionid')
    
    #keep only the options with more than 50 observationshedge_data = hedge_data[hedge_data['count']>=50]
    hedge_data = hedge_data[hedge_data['count']>=50]
    #we count them again just to know (1360)
    count_options = hedge_data.groupby(['optionid'])['date'].count()
    
    # Group the data by Option ID
    hedge_data_grouped = list(hedge_data.groupby(['optionid']))
    
    for option in hedge_data_grouped:

        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
    
        # Get the dataframe containing the option data and sort it by date
        option_data = option[1]
        option_data = option_data.sort_values(by=['date'])
        
      
        # ---------------------------- ANN -----------------------------------------
        option_data_it = option_data[
                                     [
                                     'close_norm',
                                     'strike_norm',
                                     'maturity_anunual',
                                     'interest',
                                     'volatility5',
                                     'volatility30',
                                     'volatility60',
                                     'volatility90',
                                     'volatility120',
                                     'vol_garch'
                                     ]
                                    ]   
        
        # ---------------------------- ANN -----------------------------------------       
        option_data['delta_ANN'] = np.asarray([ getDer(row, sess=sess, grad_func=grad_func, model=model) for index, row in option_data_it.iterrows()])
        option_data['delta_ANN_lag'] = option_data['delta_ANN'].shift(1)
           
        option_data['V_S_ANN'] = option_data['close_norm'] * option_data['delta_ANN']
        option_data['V_C_ANN'] = -option_data['BSgarch']
        option_data['V_B_ANN'] = getBondValueANN(option_data)   
        
        option_data['V_T_ANN'] = option_data['V_S_ANN'] + option_data['V_C_ANN'] + option_data['V_B_ANN']
            
           
        # ---------------------------- BS -----------------------------------------
        option_data['delta_BS'] = np.asarray([ getDerivativeFromSampleBS(row) for index, row in option_data.iterrows()])
        option_data['delta_BS_lag'] = option_data['delta_BS'].shift(1)
        
        option_data['V_S_BS'] = option_data['close_norm'] * option_data['delta_BS']
        option_data['V_C_BS'] = -option_data['BSgarch']
        option_data['V_B_BS'] = getBondValueBS(option_data)   
        
        option_data['V_T_BS'] = option_data['V_S_BS'] + option_data['V_C_BS'] + option_data['V_B_BS']
            
       
        # Get last row of the option_data dataframe
        last_sample = option_data.tail(1)
        
        epsilon_BS =  np.exp(-last_sample['interest']*last_sample['normmat']) * np.mean(np.abs(option_data['V_T_BS']))
        nu_BS = np.exp(-last_sample['interest']*last_sample['normmat']) * np.sqrt(np.mean(option_data['V_T_BS'])+np.var(option_data['V_T_BS']))
        epsilon_ANN = np.exp(-last_sample['interest']*last_sample['normmat']) * np.mean(np.abs(option_data['V_T_ANN']))
        nu_ANN = np.exp(-last_sample['interest']*last_sample['normmat']) * np.sqrt(np.mean(option_data['V_T_ANN'])+np.var(option_data['V_T_ANN']))
        
        performance_data_it = performance_data_it.append([{ 
                                                  'optionid': str(option[0]),
                                                  'epsilon_BS': np.asarray(epsilon_BS)[0],
                                                  'nu_BS': np.asarray(nu_BS)[0],
                                                  'epsilon_ANN': np.asarray(epsilon_ANN)[0],
                                                  'nu_ANN': np.asarray(nu_ANN)[0],
                                                  'moneyness': np.asarray(last_sample['close_norm'])[0],
                                                  'interest': np.asarray(last_sample['interest'])[0],
                                                  'maturity': np.asarray(last_sample['normmat'])[0],
                                                  'tracking_error_ANN': np.asarray(last_sample['V_T_ANN'])[0],
                                                  'tracking_error_BS': np.asarray(last_sample['V_T_BS'])[0],
                                                  }], 
                                                 ignore_index=True)
    
    
        # performance_data_it = performance_data_it.append([{ 
        #                                               'optionid': str(option[0]), 
        #                                               'moneyness': np.asarray(last_sample['close_norm'])[0],
        #                                               'interest': np.asarray(last_sample['interest'])[0],
        #                                               'maturity': np.asarray(last_sample['normmat'])[0],
        #                                               'tracking_error_ANN': np.asarray(last_sample['V_T_ANN'])[0],
        #                                               'tracking_error_BS': np.asarray(last_sample['V_T_BS'])[0],
        #                                               }], 
        #                                              ignore_index=True)
        
        #print(performance_data.to_string())
        print('INFO: ...')
        performance_data_it.to_csv('../data/performance_data_it.csv') 
        
    
    
    
    performance_data = performance_data.append([{ 'num_of_options': len(hedge_data_grouped),
                                                  'epsilon_BS': np.mean(performance_data_it['epsilon_BS']),
                                                  'nu_BS':np.mean(performance_data_it['nu_BS']),
                                                  'epsilon_ANN': np.mean(performance_data_it['epsilon_ANN']),
                                                  'nu_ANN': np.mean(performance_data_it['nu_ANN']),
                                                  'tracking_error_ANN': np.mean(performance_data_it['tracking_error_ANN']),
                                                  'tracking_error_BS': np.mean(performance_data_it['tracking_error_BS']),
                                                  }], 
                                                 ignore_index=True)
    
    #print(performance_data.to_string())
    print('INFO: Bootsrap epoch done!!')
    performance_data.to_csv('../data/performance_data.csv') 
        
        