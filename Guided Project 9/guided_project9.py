# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:16:44 2021

@author: Yash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
%matplotlib inline

data = pd.read_csv('Wholesale customers data.csv')

data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward') 
cluster = AgglomerativeClustering(distance_threshold=2, n_clusters=None, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)