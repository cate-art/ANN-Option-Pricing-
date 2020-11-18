import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

def CheckAccuracy(y, y_hat):

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
    plt.xlabel('Actual Price',fontsize=20,fontname='Times New Roman')
    plt.ylabel('Predicted Price',fontsize=20,fontname='Times New Roman') 
    plt.show()
    
    plt.figure(figsize=(14,10))
    plt.hist(diff, range=[-0.05,0.05], bins=50,edgecolor='black',color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()
    
    return stats

import pandas as pd

option = pd.read_csv('../data/index_series.csv')
call = pd.read_csv('../data/call.csv')


plot_acf(option['return'] ,lags=50)
plt.xlabel('Lags')
plt.ylabel('Correlation')
#plt.show()
plt.savefig('../images/acf.png', bbox_inches='tight')

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(option['return'] ,lags=50)
plt.xlabel('Lags')
plt.ylabel('Correlation')
#plt.show()
plt.savefig('../images/pacf.png', bbox_inches='tight')


from tabulate import tabulate
import numpy as np

stats = CheckAccuracy(call['mid'], call['BS5'])
A = stats['mae']
B = stats['mpe']
C = stats['mse']
D = stats['rmse']

stats = CheckAccuracy(call['mid'], call['BS30'])
E = stats['mae']
F = stats['mpe']
G = stats['mse']
H = stats['rmse']

stats = CheckAccuracy(call['mid'], call['BS60'])
I = stats['mae']
J = stats['mpe']
K = stats['mse']
L = stats['rmse']

stats = CheckAccuracy(call['mid'], call['BS90'])
M = stats['mae']
N = stats['mpe']
O = stats['mse']
P = stats['rmse']

stats = CheckAccuracy(call['mid'], call['BS120'])
Q = stats['mae']
R = stats['mpe']
S = stats['mse']
T = stats['rmse']

stats = CheckAccuracy(call['mid'], call['BSgarch'])
U = stats['mae']
V = stats['mpe']
W = stats['mse']
X = stats['rmse']


rows = [[' ', 'MAE', 'MPE', 'MSE', 'RMSE'],
        ['BS5',     A, B, C, D],
        ['BS30',    E, F, G, H],
        ['BS60',    I, J, K, L],
        ['BS90',    M, N, O, P],
        ['BS120',   Q, R, S, T],
        ['BSgarch', U, V, W, X]]

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))



