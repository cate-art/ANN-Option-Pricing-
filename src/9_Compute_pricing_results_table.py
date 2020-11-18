from tabulate import tabulate
import numpy as np
import pandas as pd

# ---------------------- All samples --------------------------

data = pd.read_csv('../data/pricing_data.csv') 
call = pd.read_csv('../data/call.csv') 

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

A = round(np.mean(data['mse_NN']), resolution)
B = round(np.mean(data['rmse_NN']), resolution)
C = round(np.mean(data['mae_NN']), resolution)
D = len(call)
E = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
F = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
G = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
H = round(np.mean(data['mse_BS']), resolution)
I = round(np.mean(data['rmse_BS']),resolution)
J = round(np.mean(data['mae_BS']), resolution)
K = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
L = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
M = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'
N = 0
rows = [['Model'         , 'No. Observations', 'MSE', 'RMSE', 'MAE'],
        ['Neural network',       ''          ,     A,      B,     C],
        ['CI'            ,        D          ,     E,      F,     G],
        ['Black-Scholes' ,       ''          ,     H,      I,     J],
        ['CI'            ,       ''          ,     K,      L,     M]]

print('---------------------- All samples ---------------------------')
print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))


# ---------------------- ITM samples --------------------------
data = pd.read_csv('../data/pricing_data_ITM.csv') 

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

A = round(np.mean(data['mse_NN']), resolution)
B = round(np.mean(data['rmse_NN']), resolution)
C = round(np.mean(data['mae_NN']), resolution)
D = len(call[call['moneyness'] > 1.05])
E = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
F = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
G = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
H = round(np.mean(data['mse_BS']), resolution)
I = round(np.mean(data['rmse_BS']),resolution)
J = round(np.mean(data['mae_BS']), resolution)
K = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
L = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
M = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'


# ---------------------- OTM samples --------------------------
data = pd.read_csv('../data/pricing_data_OTM.csv') 
#data.dropna(inplace=True)

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

N = round(np.mean(data['mse_NN']), resolution)
O = round(np.mean(data['rmse_NN']), resolution)
P = round(np.mean(data['mae_NN']), resolution)
Q = len(call[call['moneyness'] < 0.95])
R = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
S = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
T = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
U = round(np.mean(data['mse_BS']), resolution)
V = round(np.mean(data['rmse_BS']),resolution)
W = round(np.mean(data['mae_BS']), resolution)
X = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
Y = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
Z = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'


# ---------------------- NTM samples --------------------------
data = pd.read_csv('../data/pricing_data_NTM.csv') 
#data.dropna(inplace=True)

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

A1 = round(np.mean(data['mse_NN']), resolution)
B1 = round(np.mean(data['rmse_NN']), resolution)
C1 = round(np.mean(data['mae_NN']), resolution)
D1 = len(call[(call['moneyness'] < 1.05 )& (call['moneyness'] > 0.95)])
E1 = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
F1 = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
G1 = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
H1 = round(np.mean(data['mse_BS']), resolution)
I1 = round(np.mean(data['rmse_BS']),resolution)
J1 = round(np.mean(data['mae_BS']), resolution)
K1 = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
L1 = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
M1 = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'

rows = [['Model'         , 'Moneyness', 'No. Observations', 'MSE', 'RMSE', 'MAE'],
        ['Neural network',     ''     ,  ''          ,     A,      B,     C],
        ['CI'            ,     'ITM'  ,   D          ,     E,      F,     G],
        ['Black-Scholes' ,     ''     ,  ''          ,     H,      I,     J],
        ['CI'            ,     ''     ,  ''          ,     K,      L,     M],
        [''              ,     ''     ,  ''          ,    '',     '',    ''],
        ['Neural network',     ''     ,  ''          ,     N,      O,     P],
        ['CI'            ,     'OTM'  ,   Q          ,     R,      S,     T],
        ['Black-Scholes' ,     ''     ,  ''          ,     U,      V,     W],
        ['CI'            ,     ''     ,  ''          ,     X,      Y,     Z],
        [''              ,     ''     ,  ''          ,    '',     '',    ''],
        ['Neural network',     ''     ,  ''          ,    A1,     B1,    C1],
        ['CI'            ,     'NTM'  ,  D1          ,    E1,     F1,    G1],
        ['Black-Scholes' ,     ''     ,  ''          ,    H1,     I1,    J1],
        ['CI'            ,     ''     ,  ''          ,    K1,     L1,    M1]]

print('---------------------- ITM samples ---------------------------')
print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))



# ---------------------- ITM samples --------------------------
data = pd.read_csv('../data/pricing_data_ITM.csv') 
#data.dropna(inplace=True)

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

A = round(np.mean(data['mse_NN']), resolution)
B = round(np.mean(data['rmse_NN']), resolution)
C = round(np.mean(data['mae_NN']), resolution)
D = len(call[call['normmat'] < 1/12])
E = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
F = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
G = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
H = round(np.mean(data['mse_BS']), resolution)
I = round(np.mean(data['rmse_BS']),resolution)
J = round(np.mean(data['mae_BS']), resolution)
K = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
L = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
M = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'


# ---------------------- medium samples --------------------------
data = pd.read_csv('../data/pricing_data_medium.csv') 
#data.dropna(inplace=True)

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

N = round(np.mean(data['mse_NN']), resolution)
O = round(np.mean(data['rmse_NN']), resolution)
P = round(np.mean(data['mae_NN']), resolution)
Q = len(call[(call['normmat'] < 1/2 )& (call['normmat'] > 1/12)])
R = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
S = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
T = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
U = round(np.mean(data['mse_BS']), resolution)
V = round(np.mean(data['rmse_BS']),resolution)
W = round(np.mean(data['mae_BS']), resolution)
X = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
Y = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
Z = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'


# ---------------------- long samples --------------------------
data = pd.read_csv('../data/pricing_data_long.csv') 
#data.dropna(inplace=True)

# confidence intervals
alpha = 0.95
p1 = ((1.0-alpha)/2.0) * 100
p2 = (alpha+((1.0-alpha)/2.0)) * 100
resolution = 5

A1 = round(np.mean(data['mse_NN']), resolution)
B1 = round(np.mean(data['rmse_NN']), resolution)
C1 = round(np.mean(data['mae_NN']), resolution)
D1 = len(call[call['normmat'] > 1/2])
E1 = '(' + str(round(max(0.0, np.percentile(data['mse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_NN'], p2)), resolution)) + ')'
F1 = '(' + str(round(max(0.0, np.percentile(data['rmse_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_NN'], p2)), resolution)) + ')'
G1 = '(' + str(round(max(0.0, np.percentile(data['mae_NN'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_NN'], p2)), resolution)) + ')'
H1 = round(np.mean(data['mse_BS']), resolution)
I1 = round(np.mean(data['rmse_BS']),resolution)
J1 = round(np.mean(data['mae_BS']), resolution)
K1 = '(' + str(round(max(0.0, np.percentile(data['mse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mse_BS'], p2)), resolution)) + ')'
L1 = '(' + str(round(max(0.0, np.percentile(data['rmse_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['rmse_BS'], p2)), resolution)) + ')'
M1 = '(' + str(round(max(0.0, np.percentile(data['mae_BS'], p1)), resolution)) + ',' +str(round(min(1.0, np.percentile(data['mae_BS'], p2)), resolution)) + ')'

rows = [['Model'         , 'Maturity', 'No. Observations', 'MSE', 'RMSE', 'MAE'],
        ['Neural network',     ''     ,  ''          ,     A,      B,     C],
        ['CI'            ,   'Short'  ,   D          ,     E,      F,     G],
        ['Black-Scholes' ,     ''     ,  ''          ,     H,      I,     J],
        ['CI'            ,     ''     ,  ''          ,     K,      L,     M],
        [''              ,     ''     ,  ''          ,    '',     '',    ''],
        ['Neural network',     ''     ,  ''          ,     N,      O,     P],
        ['CI'            ,   'Medium' ,   Q          ,     R,      S,     T],
        ['Black-Scholes' ,     ''     ,  ''          ,     U,      V,     W],
        ['CI'            ,     ''     ,  ''          ,     X,      Y,     Z],
        [''              ,     ''     ,  ''          ,    '',     '',    ''],
        ['Neural network',     ''     ,  ''          ,    A1,     B1,    C1],
        ['CI'            ,    'Long'  ,  D1          ,    E1,     F1,    G1],
        ['Black-Scholes' ,     ''     ,  ''          ,    H1,     I1,    J1],
        ['CI'            ,     ''     ,  ''          ,    K1,     L1,    M1]]

print('---------------------- ITM samples ---------------------------')
print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))


