from tabulate import tabulate
import numpy as np
import pandas as pd

# ---------------------- All samples --------------------------

data = pd.read_csv('../data/performance_data.csv') 

from scipy import stats 

data['tracking_error_ANN']=np.abs(data['tracking_error_ANN'])
data['tracking_error_BS']=np.abs(data['tracking_error_BS'])

resolution = 9
data_ITM = data[data['moneyness'] > 1.05]
t_test_ITM = stats.ttest_ind_from_stats(np.mean(data_ITM['tracking_error_ANN']), 
                                        np.std(data_ITM['tracking_error_ANN']),
                                        len(data_ITM['tracking_error_ANN']), 
                                        np.mean(data_ITM['tracking_error_BS']),
                                        np.std(data_ITM['tracking_error_BS']),
                                        len(data_ITM['tracking_error_BS']))
print(t_test_ITM)
A = np.round(t_test_ITM[0], resolution)
B = np.round(t_test_ITM[1], resolution)
C = len(data_ITM)
D = np.round((len(data_ITM[(data_ITM['tracking_error_ANN'] < data_ITM['tracking_error_BS'])])/C) *100,2)
data_OTM = data[data['moneyness'] < 0.95]
t_test_OTM = stats.ttest_ind(data_OTM['tracking_error_ANN'], data_OTM['tracking_error_BS'])
print(t_test_OTM)
E = np.round(t_test_OTM[0], resolution)
F = np.round(t_test_OTM[1], resolution)
G = len(data_OTM)
H = np.round((len(data_OTM[(data_OTM['tracking_error_ANN'] < data_OTM['tracking_error_BS'])])/G) *100,2)
data_NTM = data[(data['moneyness'] < 1.05 ) & (data['moneyness'] > 0.95)]
t_test_NTM = stats.ttest_ind(data_NTM['tracking_error_ANN'], data_NTM['tracking_error_BS'])
print(t_test_NTM)
I = np.round(t_test_NTM[0], resolution)
J = np.round(t_test_NTM[1], resolution)
K = len(data_NTM)
L = np.round((len(data_NTM[(data_NTM['tracking_error_ANN'] < data_NTM['tracking_error_BS'])])/K) *100,2)
t_test = stats.ttest_ind(data['tracking_error_ANN'], data['tracking_error_BS'])
print(t_test)
M = np.round(t_test[0], resolution)
N = np.round(t_test[1], resolution)
O = len(data)
P = np.round((len(data[(data['tracking_error_ANN'] < data['tracking_error_BS'])])/O) *100,2)

rows = [['Sample' , 't-statistic', 'p-value', 'Observations', '% NN < BS'],
        ['ITM',                 A,         B,              C,           D],
        ['OTM',                 E,         F,              G,           H],
        ['NTM',                 I,         J,              K,           L],
        ['All',                 M,         N,              O,           P]]

print('---------------------- All samples ---------------------------')
print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))
