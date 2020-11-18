import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.MLP import build, loadData, splitDataXY, splitTrainAndTest, CheckAccuracy, plotTrainingMetrics

# Load the dataset
data = loadData('../data/call.csv') 
description = data.describe()

# Add a column with the index for each sample. This is used for selecting the test and training test
data['index'] = range(1, len(data) + 1)

num_of_sample = int(0.75 * len(data))
num_of_epochs = 5
num_of_bootstraping_iter = 4000
test_loss_min = 1

# Build the model
[X,Y] = splitDataXY(data)

# Container to save the results
import os.path 

# Check if the .csv file already exists
if os.path.isfile('../data/pricing_data.csv'):
     pricing_data = pd.read_csv('../data/pricing_data.csv')
     print('INFO: csv file found.')
else:
     pricing_data = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data.to_csv('../data/pricing_data.csv') 
     print('INFO: csv file not found. Initializing dataframe')
     
     
if os.path.isfile('../data/pricing_data_ITM.csv'):
     pricing_data_ITM = pd.read_csv('../data/pricing_data_ITM.csv')
     print('INFO: csv file found.')
else:
     pricing_data_ITM = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data_ITM.to_csv('../data/pricing_data_ITM.csv') 
     print('INFO: csv file not found. Initializing dataframe')
     
if os.path.isfile('../data/pricing_data_OTM.csv'):
     pricing_data_OTM = pd.read_csv('../data/pricing_data_OTM.csv')
     print('INFO: csv file found.')
else:
     pricing_data_OTM = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data_OTM.to_csv('../data/pricing_data_OTM.csv') 
     print('INFO: csv file not found. Initializing dataframe')

if os.path.isfile('../data/pricing_data_NTM.csv'):
     pricing_data_NTM = pd.read_csv('../data/pricing_data_NTM.csv')
     print('INFO: csv file found.')
else:
     pricing_data_NTM = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data_NTM.to_csv('../data/pricing_data_NTM.csv') 
     print('INFO: csv file not found. Initializing dataframe')
     
     
if os.path.isfile('../data/pricing_data_short.csv'):
     pricing_data_short = pd.read_csv('../data/pricing_data_short.csv')
     print('INFO: csv file found.')
else:
     pricing_data_short = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data_short.to_csv('../data/pricing_data_short.csv') 
     print('INFO: csv file not found. Initializing dataframe')
     
if os.path.isfile('../data/pricing_data_medium.csv'):
     pricing_data_medium = pd.read_csv('../data/pricing_data_medium.csv')
     print('INFO: csv file found.')
else:
     pricing_data_medium = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data_medium.to_csv('../data/pricing_data_medium.csv') 
     print('INFO: csv file not found. Initializing dataframe')
     
if os.path.isfile('../data/pricing_data_long.csv'):
     pricing_data_long = pd.read_csv('../data/pricing_data_long.csv')
     print('INFO: csv file found.')
else:
     pricing_data_long = pd.DataFrame(columns=['iteration', 'mse_BS', 'rmse_BS', 'mae_BS', 'mpe_BS','mse_NN', 'rmse_NN', 'mae_NN', 'mpe_NN'])
     pricing_data_long.to_csv('../data/pricing_data_long.csv') 
     print('INFO: csv file not found. Initializing dataframe')
     
     
for bootstraping_iter in range(num_of_bootstraping_iter):
    
    model = build(X)
    best_model = model
    
    # Container for the training data:
    val_loss = list()
    train_loss = list()
    test_loss = list()

    
    boot_indexes = resample(data['index'], replace=True, n_samples=num_of_sample, random_state=bootstraping_iter)
    oob_indexes  = [x for x in data['index'] if x not in boot_indexes]
    
    boot = resample(data, replace=True, n_samples=num_of_sample, random_state=bootstraping_iter)
    obb =  data.loc[data['index'].isin(oob_indexes)]
    
    #--------------  All samples --------------------
    
    [X_train, Y_train] = splitDataXY(boot)
    [X_test, Y_test]   = splitDataXY(obb)

    # history = model.fit(X_train, Y_train, batch_size=64, epochs=num_of_epochs, verbose=1)
    # eval_metrics = model.evaluate(X_test, Y_test)
    
    for i in range(num_of_epochs):    
        # Learning and Evaluation
        history = model.fit(X_train,
                            Y_train,
                            batch_size=64,
                            epochs=1,
                            verbose=1)
        # Save training metrics
        train_loss.append(history.history['loss'][0])
        
        # Save testing metrics
        eval_metrics = model.evaluate(X_test, Y_test)
        test_loss.append(eval_metrics)
        
        # Plot loss through training and testing
        plotTrainingMetrics(train_loss, test_loss)
            
        if test_loss_min > eval_metrics:
          print("INFO: Saving best candidate model with error = {0:.9f} . Previous best was: error = {1:.9f}".format(eval_metrics, test_loss_min))
          test_loss_min = eval_metrics
          best_model = model
          
          # Save the model
          model.save_weights('../models/' + 'bootstrap_model' + '_weights.h5'  )
    
    model = best_model
    Y_hat = model.predict([X_test]) 
    Y_BS  = obb['BSgarch']/obb['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data = pricing_data.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)
    
    pricing_data.to_csv('../data/pricing_data.csv') 

    #--------------  In the money samples --------------------
    
    obb_ITM = obb[(obb['moneyness'] > 1.05)]
    [X_test, Y_test]   = splitDataXY(obb_ITM)

    Y_hat = model.predict([X_test]) 
    Y_BS  = obb_ITM['BSgarch']/obb_ITM['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data_ITM = pricing_data_ITM.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)

    pricing_data_ITM.to_csv('../data/pricing_data_ITM.csv') 
    
    #--------------  out of the money samples --------------------
    
    obb_OTM = obb[obb['moneyness'] < 0.95]
    [X_test, Y_test]   = splitDataXY(obb_OTM)

    Y_hat = model.predict([X_test]) 
    Y_BS  = obb_OTM['BSgarch']/obb_OTM['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data_OTM = pricing_data_OTM.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)
   
    pricing_data_OTM.to_csv('../data/pricing_data_OTM.csv') 
       
    #--------------  near the money samples --------------------
    
    obb_NTM = obb[(obb['moneyness'] >= 0.95) & (obb['moneyness'] <= 1.05) ]
    [X_test, Y_test]   = splitDataXY(obb_NTM)

    Y_hat = model.predict([X_test]) 
    Y_BS  = obb_NTM['BSgarch']/obb_NTM['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data_NTM = pricing_data_NTM.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)
    
    # print(pricing_data_NTM.to_string())
    pricing_data_NTM.to_csv('../data/pricing_data_NTM.csv') 
    
    
    #--------------  short maturity samples --------------------
    
    obb_short = obb[(obb['normmat'] < (1/12))]
    [X_test, Y_test]   = splitDataXY(obb_short)

    Y_hat = model.predict([X_test]) 
    Y_BS  = obb_short['BSgarch']/obb_short['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data_short = pricing_data_short.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)
    
    pricing_data_short.to_csv('../data/pricing_data_short.csv') 
    
    #--------------  medium maturity samples --------------------
    
    obb_medium = obb[(obb['normmat'] >= (1/12)) & (obb['normmat'] <= (1/2))]
    [X_test, Y_test]   = splitDataXY(obb_medium)

    Y_hat = model.predict([X_test]) 
    Y_BS  = obb_medium['BSgarch']/obb_medium['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data_medium = pricing_data_medium.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)
    
    pricing_data_medium.to_csv('../data/pricing_data_medium.csv') 
    
    #--------------  long maturity samples --------------------
    
    obb_long = obb[(obb['normmat'] > (1/2))]
    [X_test, Y_test]   = splitDataXY(obb_long)

    Y_hat = model.predict([X_test]) 
    Y_BS  = obb_long['BSgarch']/obb_long['strike']
    
    diff_NN = Y_test - Y_hat[:,0]
    diff_BS = Y_test - Y_BS
    
    mse_NN = np.mean(diff_NN**2) 
    mse_BS = np.mean(diff_BS**2) 
    
    pricing_data_long = pricing_data_long.append([{ 
                                         'iteration': str(bootstraping_iter), 
                                         'mse_BS': mse_BS ,
                                         'rmse_BS': np.sqrt(mse_BS) ,
                                         'mae_BS': np.mean(abs(diff_BS)) ,
                                         'mpe_BS': np.sqrt(mse_BS)/np.mean(Y_BS),
                                         'mse_NN': mse_NN ,
                                         'rmse_NN': np.sqrt(mse_NN) ,
                                         'mae_NN': np.mean(abs(diff_NN)) ,
                                         'mpe_NN': np.sqrt(mse_NN)/np.mean(Y_hat),
                                         }], 
                                        ignore_index=True)
    
    pricing_data_long.to_csv('../data/pricing_data_long.csv') 