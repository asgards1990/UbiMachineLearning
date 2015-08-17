# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:34:47 2014

@author: Su
"""

import numpy as np
import pandas as pd
pd.set_option('chained_assignment',None)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500, 
                       max_depth=100,
                       min_samples_leaf=50,
                       max_features='auto',
                       oob_score=True,
                       verbose=0)

print 'Retrieving and cleaning data...'  
#Retrieving the raw data               
file_location='RawData/test_67_37_lastDayToGetDate_oldFile.csv'
df = pd.DataFrame.from_csv(file_location, sep=',')
print 'Data retrieved!'

#cleaning data
print 'Now cleaning...'
totalLength = len(df)
index = np.arange(totalLength)
df.index = index
del df['Ip']
del df['CountryShort']
df.fillna(0,inplace=True)
#df.loc[pd.isnull(df['paygap']),'paygap'] =0
#df.loc[pd.isnull(df['daysFromFirstPlayedDtToFirstPayDt']),
#'daysFromFirstPlayedDtToFirstPayDt'] =0
print 'Data cleaned! \nNow training...'

#randomized train and test samples.                   
df['is_train'] = np.random.uniform(0, 1, totalLength) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

#names of the features
features = df.columns[:21]

#training
y, _ = pd.factorize(train['Leaver'])
classifier.fit(train[features],y)
print "Data trained"

print "Now predicting..."
#prediction on test set
preds = classifier.predict(test[features])

#crosstabChecking 
print pd.crosstab(test['Leaver'], preds, rownames=['actual'], colnames=['preds'])

df['prediction'] =  df['Leaver']
df.loc[df['is_train']==False,'prediction'] = preds

file_destination = 'CleanData/LeaversOrNotProfiles.csv'
df.to_csv(file_destination,sep=',')

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

precision = dict()
recall = dict()
average_precision = dict()

prc = precision_recall_curve(test['Leaver'],
                                                        preds)
average_precision = average_precision_score(test['Leaver'],
                                                        preds)
                                                        
#they are in array because it computes precision and recall in relation to 0 
#and 1 respectively. The last/fourth array is the number of occurences in the
#actual data.                                                        
fscore = precision_recall_fscore_support(test['Leaver'],
                                                        preds)                                                        