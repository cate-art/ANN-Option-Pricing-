import pandas as pd

call = pd.read_csv('../data/call.csv')

from tabulate import tabulate
import numpy as np

A = np.mean(call[ (call['maturity'] < 60) & (call['moneyness'] < 0.95)]['mid'])
B = np.mean(call[ (call['maturity'] > 60) & (call['maturity'] < 180) & (call['moneyness'] < 0.95)]['mid'])
C = np.mean(call[ (call['maturity'] > 180) & (call['moneyness'] < 0.95)]['mid'])
D = np.mean(call[ (call['moneyness'] < 0.95)]['mid'])

E = np.mean(call[ (call['maturity'] < 60) & (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid'])
F = np.mean(call[ (call['maturity'] > 60) & (call['maturity'] < 180) & (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid'])
G = np.mean(call[ (call['maturity'] > 180) & (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid'])
H = np.mean(call[ (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid'])

I = np.mean(call[ (call['maturity'] < 60) & (call['moneyness'] >= 1.05)]['mid'])
J = np.mean(call[ (call['maturity'] > 60) & (call['maturity'] < 180) & (call['moneyness'] >= 1.05)]['mid'])
K = np.mean(call[ (call['maturity'] > 180) & (call['moneyness'] >= 1.05)]['mid'])
L = np.mean(call[ (call['moneyness'] >= 1.05)]['mid'])

M = (call[ (call['maturity'] < 60) & (call['moneyness'] < 0.95)]['mid']).count()
N = (call[ (call['maturity'] > 60) & (call['maturity'] < 180) & (call['moneyness'] < 0.95)]['mid']).count()
O = (call[ (call['maturity'] > 180) & (call['moneyness'] < 0.95)]['mid']).count()
P = (call[ (call['moneyness'] < 0.95)]['mid']).count()

Q = (call[ (call['maturity'] < 60) & (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid']).count()
R = (call[ (call['maturity'] > 60) & (call['maturity'] < 180) & (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid']).count()
S = (call[ (call['maturity'] > 180) & (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid']).count()
T = (call[ (call['moneyness'] > 0.95) & (call['moneyness'] < 1.05)]['mid']).count()

X = (call[ (call['maturity'] < 60) & (call['moneyness'] >= 1.05)]['mid']).count()
W = (call[ (call['maturity'] > 60) & (call['maturity'] < 180) & (call['moneyness'] >= 1.05)]['mid']).count()
Y = (call[ (call['maturity'] > 180) & (call['moneyness'] >= 1.05)]['mid']).count()
Z = (call[ (call['moneyness'] >= 1.05)]['mid']).count()


rows = [['Days to expiration', '<60', '\se 180', 'All options'],
        ['OTM(<0.95)',     A, B, C, D],
        ['ATM(0.95-1.05)', E, F, G, H],
        ['ITM(>1.05)',     I, J, K, L],
        ['OTM(<0.95)',     M, N, O, P],
        ['ATM(0.95-1.05)', Q, R, S, T],
        ['ITM(>1.05)',     X, W, Y, Z]]

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

