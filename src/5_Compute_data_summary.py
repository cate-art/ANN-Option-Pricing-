import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src.side_data_analysis as side_data_analysis
import src.xyplot_core as xyplot_core


#load needed data
pdata = pd.read_csv('../data/plot_data.csv')
description = pdata.describe()


# Import libraries 
from mpl_toolkits import mplot3d 
import numpy as np 
import matplotlib.pyplot as plt 
  
  
# Creating dataset 
pdata['price_norm'] = pdata['mid']/pdata['strike']

z = pdata['price_norm']
x = pdata['normmat']
y = pdata['moneyness']
  
# Creating figure 
fig = plt.figure(figsize = (10, 7)) 
ax = plt.axes(projection ="3d") 
ax.set_xlabel('normat')
ax.set_ylabel('moneyness')
ax.set_zlabel('price')
  
# Creating plot 
ax.scatter3D(x, y, z, color = "green"); 
plt.title("simple 3D scatter plot") 
  
# show plot 
plt.draw() 
n, bins, patches = plt.hist(pdata['moneyness'], 100, density=False, facecolor='k', alpha=0.75)

plt.axvline(x=0.95, color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=1.05, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Moneyness')
plt.ylabel('Frequency')
plt.title('Moneyness distribution')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(40, 160)
plt.ylim(0, 0.03)
plt.grid(True)
#plt.show()
plt.savefig('../images/Moneyness_distribution.png', bbox_inches='tight')

underlying = pd.read_csv ('../data/underlying.csv')
treasury = pd.read_csv ('../data/interest.csv')

#it shows the basic overview about the data structure
#plot call and put

side_data_analysis.plot_particular_option(pdata, 1400, "C")
side_data_analysis.plot_particular_option(pdata, 1300, "C")
side_data_analysis.plot_particular_option(pdata, 1200, "C")
side_data_analysis.plot_particular_option(pdata, 1100, "C")

side_data_analysis.plot_particular_option(pdata, 1400, "P")
side_data_analysis.plot_particular_option(pdata, 1300, "P")
side_data_analysis.plot_particular_option(pdata, 1200, "P")
side_data_analysis.plot_particular_option(pdata, 1100, "P")

#plot moneyness
side_data_analysis.plot_particular_moneyness(pdata,"C")
side_data_analysis.plot_particular_moneyness(pdata,"P")

#plot closing price, return and volatilities
side_data_analysis.plot_close(underlying)
side_data_analysis.plot_returns(underlying)
side_data_analysis.plot_volatilities(underlying)

#·plot the interest 
treasury['day']= range(len(treasury))
side_data_analysis.plot_interest(treasury)

#plot the black and scholes prediction
side_data_analysis.plot_black_scholes_prediction(pdata)
