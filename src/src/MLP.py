# Building the Neural Network model  MLP
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend

def loadData(path_to_file):
    # Import dataset
    data = pd.read_csv(path_to_file)
    
    # Remove NaN values due to the rolling volatility
    data = data.dropna()
    
    #normalizing the stike and price
    data['close_norm'] = data['close']/data['strike']
    data['strike_norm'] = data['strike']/data['strike']
    data['price_norm'] = data['mid']/data['strike']
    data['maturity_anunual'] = data['maturity']/252
    
    return data


def splitDataXY(data):        
    # Split the test and the training set
    """
    X = data[['strike_norm',
              'maturity']]
    """
    X = data[['close_norm',
              'strike_norm',
              'maturity_anunual',
              'interest',
              'volatility5',
              'volatility30',
              'volatility60',
              'volatility90',
              'volatility120',
              'vol_garch']]
  
    """
        X = data[['close_norm',
              'strike_norm',
              'maturity_anunual',
              'interest',
              'volatility30',
              'volatility60',
              'volatility90',
              'volatility120',
              'vol_garch',
              'norm_lag_close']]
    """
    
    #[X, _, _] = getScaledArray(np.asarray(X), high=1.0, low=0.0, bycolumn=True)
    #X =  pd.DataFrame(X)
    
    Y = data['price_norm']

    return X, Y

def splitTrainAndTest(X, Y, test_size):     
    data_split_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return data_split_test

def custom_activation(x):
    return backend.exp(x)

def build(X):    
    nodes = 120
    model = Sequential()
    
    model.add(Dense(nodes, input_dim=X.shape[1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    
    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1))
    model.add(Activation(custom_activation))
              
    model.compile(loss='mse',optimizer='rmsprop')
    
    model.summary()
    return model

def CheckAccuracy(y, y_hat, label=None):

    stats = dict()
    diff = y - y_hat
    
    stats['mse'] = np.mean(diff**2)
    print("Mean Squared Error:      ", stats['mse'])
    
    stats['rmse'] = np.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])
    
    stats['mae'] = np.mean(abs(diff))
    print("Mean Absolute Error:     ", stats['mae'])
    
    stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])
    
    #plots
    matplotlib.rcParams['agg.path.chunksize'] = 100000
    plt.figure(figsize=(14,10))
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.plot([0, 0], [0.6, 0.6], 'r-')
    plt.xlabel('Actual Price',fontsize=30,fontname='Times New Roman')
    plt.ylabel('Predicted Price',fontsize=30,fontname='Times New Roman') 
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    if label is None:
        plt.show()
    else:
        plt.savefig('../images/'+ label + '_Actual_vs_predicted_price.png', bbox_inches='tight')
    
    plt.figure(figsize=(14,10))
    plt.hist(diff, range=[-0.05,0.05], bins=50,edgecolor='black',color='white')
    plt.xlabel('Diff',fontsize=30,fontname='Times New Roman')
    plt.ylabel('Density',fontsize=30,fontname='Times New Roman')
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    if label is None:
        plt.show()
    else:
        plt.savefig('../images/' + label + '_Prediction_error_density.png', bbox_inches='tight')
        
    return stats

def plotTrainingMetrics(train_loss, test_loss):
    matplotlib.rcParams['agg.path.chunksize'] = 100000
    plt.figure(figsize=(14,10))
    plt.plot(train_loss, 'r-', label='Train')
    #plt.plot(val_loss, 'b-', label = 'Val')
    plt.plot(test_loss, 'k-', label = 'Test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch',fontsize=20,fontname='Times New Roman')
    plt.ylabel('Loss',fontsize=20,fontname='Times New Roman') 
    plt.show()

def jacobian_tensorflow(x, model, sess, num_of_outputs):    
    jacobian_matrix = []
    grad_func = tf.gradients(model.output[:, 0], model.input)
    gradients = sess.run(grad_func, feed_dict={model.input: x.reshape((1, x.size))})
    jacobian_matrix.append(gradients[0][0,:])
        
    return np.array(jacobian_matrix)

def is_jacobian_correct(model, jacobian_fn, ffpass_fn, input_samples):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    """ 
    Check of the Jacobian using numerical differentiation
    """
    num_of_inputs = 9
    #x = np.random.random((num_of_inputs,))
 
    epsilon = 1e-5   
    """ 
    Check a few columns at random
    """
    for s in np.random.choice(10, 10, replace=False): 
        x = np.asarray(input_samples.iloc[s,:])
        #for idx in np.random.choice(N, 5, replace=False): 
        for idx in range(num_of_inputs):
            x2 = x.copy()
            x2[idx] += epsilon 
            num_jacobian = (ffpass_fn(model, x2) - ffpass_fn(model, x)) / epsilon
            computed_jacobian = jacobian_fn(x, model, sess, num_of_outputs=1)
            
            diff = computed_jacobian[:, idx] - num_jacobian
            lim  = 1e-1
            print("Jacobian funtion: error: " + str(abs(diff)) + " lim: " + str(lim) + "  difference: " + str(abs(diff) - lim))
            if not all(abs(diff) < lim): 
                return False    
    return True

def ffpass_tf(model, x):
    """
    The feedforward function of our neural net
    """    
    xr = x.reshape((1, x.size))
    return model.predict(xr)[0]

#COMPUTE DERIVATIVE FOR ANN
def derivative_ANN(x, sess, model):

    J =  jacobian_tensorflow(x, model, sess, num_of_outputs=1)
    return J[0][0]


def getBondValueANN(option_data):
    tau = 1
    is_first = True
    V_B_ANN = list()
    V_B_previous = 0
    for index, row in option_data.iterrows():
    
        if is_first is True:
            V_B_it = -(row['V_S_ANN']+row['V_C_ANN'])
            V_B_ANN.append(V_B_it)
            V_B_previous = V_B_it
            is_first =  False
            continue
            
        term_A_1 = np.exp(row['interest']*tau)
        term_A_2 = V_B_previous
        term_A = term_A_1 * term_A_2
        term_B = row['close_norm']*(row['delta_ANN']-row['delta_ANN_lag'])
        V_B_it = term_A - term_B
        V_B_ANN.append(V_B_it)
        V_B_previous = V_B_it
        
    return np.array(V_B_ANN)

def getBondValueBS(option_data):
    tau = 1
    is_first = True
    V_B_BS = list()
    V_B_previous = 0
    for index, row in option_data.iterrows():
    
        if is_first is True:
            V_B_it = -(row['V_S_BS']+row['V_C_BS'])
            V_B_BS.append(V_B_it)
            V_B_previous = V_B_it
            is_first =  False
            continue
        
        term_A = np.exp(row['interest']*tau)*V_B_previous
        term_B = row['close_norm']*(row['delta_BS']-row['delta_BS_lag'])
        V_B_it = term_A - term_B
        V_B_BS.append(V_B_it)
        V_B_previous = V_B_it
        
    return np.array(V_B_BS)

def getDer(row, sess, grad_func, model):
    
    x  = np.asarray([(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)])

    x[0,0] = row['close_norm']
    x[0,1] = row['strike_norm']
    x[0,2] = row['maturity_anunual']
    x[0,3] = row['interest']
    x[0,4] = row['volatility5']
    x[0,5] = row['volatility30']
    x[0,6] = row['volatility60']
    x[0,7] = row['volatility90']
    x[0,8] = row['volatility120']
    x[0,9] = row['vol_garch']

    gradients = sess.run(grad_func, feed_dict={model.input: x.reshape((1, x.size))})
    return np.array(gradients[0][0,:])[0]

def BS_d1(S, r, tau, sigma):
    """Auxiliary function for BS model"""
    d1 = (np.log(S)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return(d1)      

def getDerivativeFromSampleBS(row):    
    d1 = BS_d1(row['close_norm'], row['interest'], row['maturity'], row['vol_garch'])
    delta_BS = scipy.stats.norm.cdf(d1)
    return delta_BS