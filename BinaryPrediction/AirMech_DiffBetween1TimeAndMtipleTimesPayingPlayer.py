# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 14:53:42 2014

@author: su.yang
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500, 
                       max_depth=100,
                       min_samples_leaf=50,
                       max_features='auto',
                       oob_score=True,
                       verbose=0)
                       
print 'Retrieving and cleaning data...'     
#Retrieving the raw data               
file_location='RawData/AirMech_DiffBetween1TimeAndMtipleTimesPayingPlayer.csv'
df = pd.DataFrame.from_csv(file_location, sep=',')
print 'Data retrieved!'

#cleaning data
print 'Now cleaning...'
totalLength = len(df)
index = np.arange(totalLength)
df.index = index
del df['Ip']
del df['CountryShort']
df.loc[pd.isnull(df['lastPayGapBeforeSecondPayDayOrLastPlayDay']),
'lastPayGapBeforeSecondPayDayOrLastPlayDay'] =0
print 'Data cleaned! \nNow training...'

#randomized train and test samples.                   
df['is_train'] = np.random.uniform(0, 1, totalLength) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

#names of the features
features = df.columns[:20]

#training
y, _ = pd.factorize(train['isMultipleTimesPayingPlayer'])
classifier.fit(train[features],y)
print "Data trained"

print "Now predicting..."
#prediction on test set
preds = classifier.predict(test[features])

#crosstabChecking 
print pd.crosstab(test['isMultipleTimesPayingPlayer'], preds, rownames=['actual'], colnames=['preds'])

df['prediction'] =  df['isMultipleTimesPayingPlayer']
df.loc[df['is_train']==False,'prediction'] = preds

file_destination = 'CleanData/oneTimeAndMultipleTimesPayerProfiles.csv'
df.to_csv(file_destination,sep=',')

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

precision = dict()
recall = dict()
average_precision = dict()

prc = precision_recall_curve(test['isMultipleTimesPayingPlayer'],
                                                        preds)
average_precision = average_precision_score(test['isMultipleTimesPayingPlayer'],
                                                        preds)
                                                        
#they are in array because it computes precision and recall in relation to 0 
#and 1 respectively. The last/fourth array is the number of occurences in the
#actual data.                                                        
fscore = precision_recall_fscore_support(test['isMultipleTimesPayingPlayer'],
                                                        preds)    