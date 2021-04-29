# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:18:27 2021

@author: Yash
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('store_data.csv', header = None, keep_default_na=False)
transactions = []

for i in range(0, 7501):
    transaction = []
    for j in range(0, 20):
        if dataset.values[i,j]!="":            
            transaction.append(str(dataset.values[i,j]))
    transactions.append(transaction)

    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)