# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:34:47 2014

@author: Su
"""

import numpy as np
#import pylab as pl
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1000, 
                       max_depth=100,
                       min_samples_leaf=50,
                       max_features='auto',
                       oob_score=True,
                       verbose=0)

#Having the raw data needed                   
file_location="C:/Users/corpus/Desktop/UbiMachineLearning/Ubi100KLeaverFor7Days3DaysBufferOutcomeInversed.csv"
df = pd.DataFrame.from_csv(file_location, sep=';')
totalLength = len(df)
index = np.arange(totalLength)
df.index = index

#cleaning data
df.index = index
del df['Ip']
del df['FirstPayDt']
del df['FirstPlayedDt']
del df['CountryShort']
del df['MaxDayMinusThree']
del df['totalUSDMaxDayMinusThree']
del df['USDMaxDayMinusThree']
for i in range (totalLength):
    if pd.isnull(df['paygap'][i]):
        df['paygap'][i] =0

#randomized train and test samples.                   
df['is_train'] = np.random.uniform(0, 1, totalLength) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

#names of the features
features = df.columns[:16]

#training
y, _ = pd.factorize(train['stopPlayingBeforeTheDay'])
classifier.fit(train[features],y)
#prediction on test set
preds = classifier.predict(test[features])

#crosstabChecking 
pd.crosstab(test['stopPlayingBeforeTheDay'], preds, rownames=['actual'], colnames=['preds'])


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

precision = dict()
recall = dict()
average_precision = dict()

prc = precision_recall_curve(test['stopPlayingBeforeTheDay'],
                                                        preds)
average_precision = average_precision_score(test['stopPlayingBeforeTheDay'],
                                                        preds)
                                                        
test = precision_recall_fscore_support(test['stopPlayingBeforeTheDay'],
                                                        preds)                                                        