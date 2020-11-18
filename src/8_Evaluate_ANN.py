import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd


from src.MLP import build, loadData, splitDataXY, splitTrainAndTest, CheckAccuracy

# Load the dataset
data = loadData('../data/call.csv') 
description = data.describe()

# Get Input and Output data
[X, Y] = splitDataXY(data)

# Get Training, Evaluation and Test splits
[X_train, X_test, y_train, y_test] = splitTrainAndTest(X, Y, test_size=0.25)

# Get the data corresponding to the test set
[_, _, _, y_Date] = splitTrainAndTest(X, data['date'], test_size=0.25)
y_Date = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in y_Date]
[_, _, _, y_BS] = splitTrainAndTest(X, data['BSgarch']/data['strike'], test_size=0.25)
[_, _, _, y_moneyness] = splitTrainAndTest(X, data['moneyness'], test_size=0.25)
[_, _, _, y_maturity] = splitTrainAndTest(X, data['maturity_anunual'], test_size=0.25)
[_, _, _, y_volatility] = splitTrainAndTest(X, data['vol_garch'], test_size=0.25)

# Build the model
model = build(X_train)
base_filename = 'bootstrap_model'
model.load_weights('../models/' + base_filename + '_weights.h5')

# Get the predicted values for the test set
y_hat = model.predict([X_test])  

# Compare real values of the test set vs Black and scholes and The Neural network
CheckAccuracy(y_test, y_BS, label='ANN_eval_BS')
CheckAccuracy(y_test, y_hat[:,0], label='ANN_eval_NN')

#plot moneyness
matplotlib.rcParams['agg.path.chunksize'] = 100000
plt.figure(figsize=(14,10))
pricing_error_BS = (y_test - y_BS)
plt.scatter(pricing_error_BS, y_moneyness, color='black',linewidth=0.3,alpha=0.4, s=0.5)
plt.plot([np.min(pricing_error_BS), np.max(pricing_error_BS)], [1, 1], 'r-')
plt.xlabel('Pricing Error (BS)',fontsize=30,fontname='Times New Roman')
plt.ylabel('Moneyness',fontsize=30,fontname='Times New Roman')
plt.xlim(np.min(pricing_error_BS), np.max(pricing_error_BS))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_BS_Moneyness_vs_Pricing_error.png', bbox_inches='tight')

matplotlib.rcParams['agg.path.chunksize'] = 100000
plt.figure(figsize=(14,10))
pricing_error_NN = (y_test - y_hat[:,0])
plt.scatter(pricing_error_NN, y_moneyness, color='black',linewidth=0.3,alpha=0.4, s=0.5)
plt.plot([np.min(pricing_error_BS), np.max(pricing_error_BS)], [1, 1], 'r-')
plt.xlabel('Pricing Error (NN)',fontsize=30,fontname='Times New Roman')
plt.ylabel('Moneyness',fontsize=30,fontname='Times New Roman')
plt.xlim(np.min(pricing_error_BS), np.max(pricing_error_BS))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_NN_Moneyness_vs_Pricing_error.png', bbox_inches='tight')

plt.figure(figsize=(14,10))
plt.hist(y_moneyness, range=[0.5, 1.6], bins=50,edgecolor='black',color='white')
plt.xlabel('Moneyness',fontsize=30,fontname='Times New Roman')
plt.ylabel('Density',fontsize=30,fontname='Times New Roman')
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_moneyness_density.png', bbox_inches='tight')

#plot maturity
matplotlib.rcParams['agg.path.chunksize'] = 100000
plt.figure(figsize=(14,10))
pricing_error_BS = (y_test - y_BS)
plt.scatter(pricing_error_BS, y_maturity, color='black',linewidth=0.3,alpha=0.4, s=0.5)
plt.xlabel('Pricing Error (BS)',fontsize=30,fontname='Times New Roman')
plt.ylabel('Maturity',fontsize=30,fontname='Times New Roman')
plt.xlim(np.min(pricing_error_BS), np.max(pricing_error_BS))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_BS_Maturity_vs_Pricing_error.png', bbox_inches='tight')

matplotlib.rcParams['agg.path.chunksize'] = 100000
plt.figure(figsize=(14,10))
pricing_error_NN = (y_test - y_hat[:,0])
plt.scatter(pricing_error_NN, y_maturity, color='black',linewidth=0.3,alpha=0.4, s=0.5)
plt.xlabel('Pricing Error (NN)',fontsize=30,fontname='Times New Roman')
plt.ylabel('Maturity',fontsize=30,fontname='Times New Roman')
plt.xlim(np.min(pricing_error_BS), np.max(pricing_error_BS))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_NN_Maturity_vs_Pricing_error.png', bbox_inches='tight')

plt.figure(figsize=(14,10))
plt.hist(y_maturity, range=[0,2], bins=50,edgecolor='black',color='white')
plt.xlabel('Maturity',fontsize=30,fontname='Times New Roman')
plt.ylabel('Density',fontsize=30,fontname='Times New Roman')
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_maturity_density.png', bbox_inches='tight')

#plot volatility 90
matplotlib.rcParams['agg.path.chunksize'] = 1000009
plt.figure(figsize=(14,10))
pricing_error_BS = (y_test - y_BS)
plt.scatter(pricing_error_BS, y_volatility, color='black',linewidth=0.3,alpha=0.4, s=0.5)
plt.xlabel('Pricing Error (BS)',fontsize=30,fontname='Times New Roman')
plt.ylabel('Volatility GARCH',fontsize=30,fontname='Times New Roman')
plt.xlim(np.min(pricing_error_BS), np.max(pricing_error_BS))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_BS_Volatility_vs_Pricing_error.png', bbox_inches='tight')

matplotlib.rcParams['agg.path.chunksize'] = 100000
plt.figure(figsize=(14,10))
pricing_error_NN = (y_test - y_hat[:,0])
plt.scatter(pricing_error_NN, y_volatility, color='black',linewidth=0.3,alpha=0.4, s=0.5)
plt.xlabel('Pricing Error (NN)',fontsize=30,fontname='Times New Roman')
plt.ylabel('Volatility GARCH',fontsize=30,fontname='Times New Roman')
plt.xlim(np.min(pricing_error_BS), np.max(pricing_error_BS))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_NN_Volatility_vs_Pricing_error.png', bbox_inches='tight')

plt.figure(figsize=(14,10))
plt.hist(y_volatility, range=[0,0.6], bins=50,edgecolor='black',color='white')
plt.xlabel('Volatility GARCH',fontsize=30,fontname='Times New Roman')
plt.ylabel('Density',fontsize=30,fontname='Times New Roman')
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_Volatility_Density.png', bbox_inches='tight')

#plot price density
plt.figure(figsize=(14,10))
plt.hist(y_test, range=[0,0.60], bins=50,edgecolor='black',color='white')
plt.xlabel('Price',fontsize=30,fontname='Times New Roman')
plt.ylabel('Density',fontsize=30,fontname='Times New Roman')
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
#plt.show()
plt.savefig('../images/ANN_eval_Price_Density.png', bbox_inches='tight')
    
"""
fig, ax = plt.subplots()
ax.plot(y_Date, y_test, "ob")     
ax.plot(y_Date, y_BS, "or")  
ax.plot(y_Date, y_hat, "og") 

ax.set(xlabel='time (s)', 
       ylabel='prices',
       title='Prices: Predicted vs Real')
ax.grid()
fig.savefig("test.png")
plt.show()
"""