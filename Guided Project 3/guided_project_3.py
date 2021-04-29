# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:34:34 2021

@author: Dell
"""

import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


df= pd.read_csv("train.csv")

df.columns
df.drop(['Gender', 'id', 'Unnamed: 0'],axis=1,inplace=True)
df.dropna(inplace=True)

labelEncoder_X = LabelEncoder()
df['Customer Type'] = labelEncoder_X.fit_transform(df['Customer Type'])
df['Type of Travel'] = labelEncoder_X.fit_transform(df['Type of Travel'])
df['Class'] = labelEncoder_X.fit_transform(df['Class'])
df['satisfaction'] = labelEncoder_X.fit_transform(df['satisfaction'])

fa = FactorAnalyzer()
fa.set_params(n_factors=6, rotation='varimax')
fa.fit(df)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

fa.loadings_
fa.get_factor_variance()