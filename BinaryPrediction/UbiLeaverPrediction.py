# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:34:47 2014

@author: Su
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1000, 
                       max_depth=100,
                       min_samples_leaf=50,
                       max_features='auto',
                       oob_score=True,
                       verbose=0)

#Retrieving the raw data               
file_location='RawData/Ubi100KFileLeaverFor60Days1WeekBufferFromLevel2.csv'
df = pd.DataFrame.from_csv(file_location, sep=',')


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

#randomized train and test samples.                   
df['is_train'] = np.random.uniform(0, 1, totalLength) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

#names of the features
features = df.columns[:21]

#training
y, _ = pd.factorize(train['stopPlayingBeforeTheDayAtLeastLevel2'])
classifier.fit(train[features],y)
#prediction on test set
preds = classifier.predict(test[features])

#crosstabChecking 
pd.crosstab(test['stopPlayingBeforeTheDayAtLeastLevel2'], preds, rownames=['actual'], colnames=['preds'])


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

precision = dict()
recall = dict()
average_precision = dict()

prc = precision_recall_curve(test['stopPlayingBeforeTheDayAtLeastLevel2'],
                                                        preds)
average_precision = average_precision_score(test['stopPlayingBeforeTheDayAtLeastLevel2'],
                                                        preds)
                                                        
#they are in array because it computes precision and recall in relation to 0 
#and 1 respectively. Teh last/fourth array is the number of occurences in the
#actual data.                                                        
fscore = precision_recall_fscore_support(test['stopPlayingBeforeTheDayAtLeastLevel2'],
                                                        preds)                                                        