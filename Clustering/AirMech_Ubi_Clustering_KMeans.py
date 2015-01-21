# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 08:48:13 2014

@author: su.yang
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import Counter

#Retrieving the raw data 
print 'Retrieving and cleaning data...'              
file_location='RawData/Ubi100KFileLeaverFor60Days1WeekBufferFromLevel2.csv'
df = pd.DataFrame.from_csv(file_location, sep=',')
print 'Data retrieved!'

#cleaning data
totalLength = len(df)
index = np.arange(totalLength)
df.index = index
del df['Ip']
del df['CountryShort']
for i in range (totalLength):
    if pd.isnull(df['paygap'][i]):
        df['paygap'][i] =0
    if pd.isnull(df['daysFrom20140718ToFirstPayDt'][i]):
        df['daysFrom20140718ToFirstPayDt'][i]=0

print 'Data cleaned! \nNow training...'

kmeans_model = KMeans(n_clusters=10, random_state=1).fit(df)
labels = kmeans_model.labels_
#metrics.silhouette_score(df, labels, metric='euclidean')

print 'Training is over, here the label distribution: '
print Counter(labels)
df['labels'] = labels
setZero = df[df['labels']==0]
setOne = df[df['labels']==1]

file_destination = 'CleanData/LeaverProfiles'
df.to_csv(file_destination,sep=',')

