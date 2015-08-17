# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:34:47 2014

@author: Su
"""

import numpy as np
import pandas as pd
import os
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
pd.set_option('chained_assignment',None)
classifier=RandomForestClassifier(
                        n_estimators=250, 
                        max_depth=8,
                        min_samples_leaf=5,
                        max_features='auto',
                        oob_score=True,
                        verbose=0)

os.chdir(os.path.dirname(os.path.abspath('__file__')))
print 'Retrieving and cleaning data...'  
#Retrieving the raw data               
file_location='RawData/first23DaysOldGen_20150526.csv'
print file_location
df = pd.DataFrame.from_csv(file_location, sep=',')
file_location = 'RawData/first23DaysNewGen_20150526.csv'
target = pd.DataFrame.from_csv(file_location, sep=',')

print 'Data retrieved!'

#cleaning data
print 'Now cleaning...'
totalLength = len(df)
index = np.arange(totalLength)
df.index = index
#del df['Ip']

print 'Quantifying CountryCode...'
df['CountryShort'].fillna('nan',inplace=True)
target['CountryShort'].fillna('nan',inplace=True)
lbl = preprocessing.LabelEncoder()
lbl.fit(list(df['CountryShort'].values)+list(target['CountryShort']))
df['CountryShort'] = lbl.transform(df['CountryShort'].values)
target['CountryShort'] = lbl.transform(target['CountryShort'].values)
target.fillna(-1,inplace=True)
df.fillna(-1,inplace=True)
print 'Replacing Nas and deleting unecessary columns...'

del df['paygap']
del target['paygap']
del df['USDMaxDay']
del target['USDMaxDay']
#del df['NonPersistentLevel']
#del target['NonPersistentLevel']
del df['averageAirMechExpPerTime']
del target['averageAirMechExpPerTime']


print 'Data cleaned! \nNow training...'
#randomized train and test samples.                   
df['is_train'] = np.random.uniform(0, 1, totalLength) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

#names of the features, the labels and is_train need to be excluded.
features = df.columns[:df.shape[1]-2]

#training
y, _ = pd.factorize(train['Leaver'])
classifier.fit(train[features],y)
print "Data trained\n"

print "Now predicting..."
#prediction on test set
preds = classifier.predict(test[features])

#crosstabChecking 
print pd.crosstab(test['Leaver'], preds, rownames=['actual'], colnames=['preds'])

df['prediction'] =  df['Leaver']
df.loc[df['is_train']==False,'prediction'] = preds



from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

precision = dict()
recall = dict()
average_precision = dict()

prc = precision_recall_curve(test['Leaver'],preds)
average_precision = average_precision_score(test['Leaver'],preds)
                                                        
#they are in arrayay because it computes precision and recall in relation to 0 
#and 1 respectively. The last/fourth array is the number of occurences in the
#actual data.                                                        
fscore = precision_recall_fscore_support(test['Leaver'],
                                                        preds)                                                        
                                                        
#predictions = classifier.predict_proba(train[features])[:,1]
#print roc_auc_score(y,predictions)

print fscore
#
#file_destination = 'CleanData/LeaversOrNotProfiles.csv'
#df.to_csv(file_destination,sep=',')
print train[features].columns.values
weights=classifier.feature_importances_
columns=list(train[features].columns.values)
for i in range(len(columns)):
    print columns[i]
    print weights[i]
weights = pd.DataFrame(weights)
weights.columns=[1]
weights = pd.concat([pd.DataFrame(features),weights], axis=1)

predTarget=classifier.predict(target[features])
target['prediction']=predTarget
file_destination = 'CleanData/newGenPrediction.csv'
target.to_csv(file_destination,sep=',')


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=(0,1), cv=3,n_jobs=1, train_sizes=range(1000,7000,1000)):
    """
    Generate a simple plot of the test and traning learning curve.
    
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.
    
    title : string
    Title for the chart.
    
    X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
    
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.
    
    ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.
    
    cv : integer, cross-validation generator, optional
    If an integer is passed, it is the number of folds (defaults to 3).
    Specific cross-validation objects can be passed, see
    sklearn.cross_validation module for the list of possible objects
    
    n_jobs : integer, optional
    Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    #learning_curve(estimator, X, y,train_sizes=train_sizes)
    train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X, y,scoring='roc_auc',train_sizes=train_sizes) 
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
#    
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

plot_learning_curve(classifier,'learning curve',train[features],y)