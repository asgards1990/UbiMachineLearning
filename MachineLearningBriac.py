# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:31:28 2014

@author: bpiriou, aubertlebrozec
"""

import numpy as np

#Data
import dataParser

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Deep Learning
from nolearn.dbn import DBN
from gdbn.activationFunctions import Sigmoid
#from gdbn.activationFunctions import Softmax
#from gdbn.activationFunctions import Linear

#SVM
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

#KNeighbors
from sklearn.neighbors import KNeighborsClassifier

#GridSearch
from sklearn.grid_search import GridSearchCV

#gestion fichiers
from sklearn.externals import joblib
import os

#from scipy.stats import itemfreq


#parameter function
def parameterFunction(model):
    if (model == 'RF'):
        param_grid = [{'n_estimators':[1000], 
                       'max_depth':[100],
                       'min_samples_leaf':[50],
                       'max_features':['auto'],
                       'oob_score':[True],
                       'verbose':[0]},
        ]
    if (model == 'NN'): 
        param_grid = [{'layer_sizes':[[-1,128,-1]], 
                       'learn_rates':[0.0001,0.001,0.01,0.1],
                       'learn_rate_decays':[0.9,0.95,1],
                       'learn_rates_pretrain':[0.001,0.01],
                       'epochs':[200],
                       'output_act_funct':[Sigmoid()],
                       'verbose':[0]},
        ]
    if (model == 'SVM'):
        param_grid = [{'alpha':[0.005],
                       'loss':['modified_huber'],
                       'penalty':['l1'],
                       'n_iter':[3000],
                       'verbose':[0]},
        ]
    if (model == 'SVM2'):
        param_grid = [{'C':[0.001,0.01,0.1,1.,10.,100.],
                       'gamma':[0.0001,0.001,0.01],
                       'cache_size':[2000],
                       'kernel':['rbf'],
                       'probability':[True],
                       'verbose':[0]},
        ]
    if (model=='KN'):
        param_grid = [{'n_neighbors':[200,300],
                       'weights':['distance'],
                       'metric':['minkowski']},
        ]
    if (model == 'Fusion'):
        param_grid = [{'n_estimators':[1000], 
                       'max_depth':[10,20,50,100],
                       'min_samples_leaf':[500,1000,2000],
                       'max_features':['auto'],
                       'oob_score':[True],
                       'verbose':[0]},
        ]
    return param_grid  
        
#learning functions
def learn(typeAlgo,train,targetTrain,fileName):
    
    createFolders(typeAlgo)
    
    folder=typeAlgo+'//clf//'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder,the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e
            
    #parameters        
    param_grid = parameterFunction(typeAlgo)
    
    #classifier
    if (typeAlgo=='RF') :
        classifier=RandomForestClassifier()
    elif (typeAlgo=='NN') :
        classifier=DBN()
    elif (typeAlgo=='SVM'):
        classifier=SGDClassifier()
    elif (typeAlgo=='SVM2'):
        classifier=SVC()
    elif (typeAlgo=='KN'):
        classifier=KNeighborsClassifier()
    elif (typeAlgo=='Fusion'):
        classifier=RandomForestClassifier()
        

        
        predict('RF',train,'learningRF.pk1','fusionTrain2')
        predict('NN',train,'learningNN.pk1','fusionTrain2')
        predict('SVM',train,'learningSVM.pk1','fusionTrain2')
        predict('SVM2',train,'learningSVM2.pk1','fusionTrain2') 
        predict('KN',train,'learningKN.pk1','fusionTrain2')  
        proba_train2RF = np.load('RF//probaRFfusionTrain2.txt')
        proba_train2NN = np.load('NN//probaNNfusionTrain2.txt')
        proba_train2SVM = np.load('SVM//probaSVMfusionTrain2.txt')
        proba_train2SVM2 = np.load('SVM2//probaSVM2fusionTrain2.txt')
        proba_train2KN = np.load('KN//probaKNfusionTrain2.txt')    
        
        Train2f=np.append(proba_train2RF[:,1],proba_train2NN[:,1])
        Train2f=np.append(Train2f,proba_train2SVM[:,1])
        Train2f=np.append(Train2f,proba_train2SVM2[:,1])
        Train2f=np.append(Train2f,proba_train2KN[:,1])
        Train2f=np.reshape(Train2f,(5,np.shape(proba_train2RF)[0]))
        train=np.transpose(Train2f)
    
    #gridcv    
    gridCV=GridSearchCV(classifier,param_grid,cv=3,scoring='accuracy')
    gridCV.fit(train,targetTrain)
    
    #best estimator
    clf=gridCV.best_estimator_
    clf.fit(train,targetTrain)
    
    params=gridCV.best_params_
    score=gridCV.best_score_
    
    #dump files
    joblib.dump(clf,folder+fileName) 
    
    f=open(typeAlgo+'//paramsOptimisation'+typeAlgo+'.txt',"w") 
    for i in params:
        f.write(i+'='+str(params[i])+"\n")
    f.close()
    
    f=open(typeAlgo+'//score'+typeAlgo+'_crossValidation.txt',"w") 
    f.write(str(score)+"\n")
    f.close()
    


#predict functions
def predict(typeAlgo,test,fileName,svgName):
    folder=typeAlgo+'//clf//'
    clf = joblib.load(folder+fileName)
 
    

    if (typeAlgo=='RF'):
        featImp=clf.feature_importances_      
    elif (typeAlgo=='SVM'):
        featImp=clf.coef_  
    elif (typeAlgo=='Fusion'):
        featImp=clf.feature_importances_
        
        predict('RF',test,'learningRF.pk1','fusionTest2')
        predict('NN',test,'learningNN.pk1','fusionTest2')
        predict('SVM',test,'learningSVM.pk1','fusionTest2')
        predict('SVM2',test,'learningSVM2.pk1','fusionTest2') 
        predict('KN',test,'learningKN.pk1','fusionTest2')  
        proba_test2RF = np.load('RF//probaRFfusionTest2.txt')
        proba_test2NN = np.load('NN//probaNNfusionTest2.txt')
        proba_test2SVM = np.load('SVM//probaSVMfusionTest2.txt')
        proba_test2SVM2 = np.load('SVM2//probaSVM2fusionTest2.txt')
        proba_test2KN = np.load('KN//probaKNfusionTest2.txt')    
        
        Test2f=np.append(proba_test2RF[:,1],proba_test2NN[:,1])
        Test2f=np.append(Test2f,proba_test2SVM[:,1])
        Test2f=np.append(Test2f,proba_test2SVM2[:,1])
        Test2f=np.append(Test2f,proba_test2KN[:,1])
        Test2f=np.reshape(Test2f,(5,np.shape(proba_test2RF)[0]))
        test=np.transpose(Test2f)
    else:
        featImp=0

    y_pred=clf.predict(test)    
    y_pred_proba=clf.predict_proba(test)
        
    #write .txt    
    f=open(typeAlgo+'//featImportance'+typeAlgo+svgName+'.txt',"w") 
    np.save(f,featImp)
    f.close()
    
    f=open(typeAlgo+'//y'+typeAlgo+svgName+'.txt',"w") 
    np.save(f,y_pred)
    f.close()
    
    f=open(typeAlgo+'//proba'+typeAlgo+svgName+'.txt',"w")
    np.save(f,y_pred_proba)
    f.close()     
    
   
def FusionLightpredict(train2,test2,targetTrain2,targetTest2):   
    
    createFolders('FusionLight')    
    
    fusionLightTest=np.vstack((train2,test2))
    fusionLightTargetTest=np.append(targetTrain2,targetTest2)
    predict('RF',fusionLightTest,'learningRF.pk1','fusionLightTest')
    predict('NN',fusionLightTest,'learningNN.pk1','fusionLightTest')    
    predict('SVM',fusionLightTest,'learningSVM.pk1','fusionLightTest')
    predict('SVM2',fusionLightTest,'learningSVM2.pk1','fusionLightTest') 
    predict('KN',fusionLightTest,'learningKN.pk1','fusionLightTest')  
    
    proba_predRF = np.load('RF//probaRFfusionLightTest.txt')
    proba_predNN = np.load('NN//probaNNfusionLightTest.txt')
    proba_predSVM = np.load('SVM//probaSVMfusionLightTest.txt')
    proba_predSVM2 = np.load('SVM2//probaSVM2fusionLightTest.txt')
    proba_predKN = np.load('KN//probaKNfusionLightTest.txt')    
        
    y_pred_fusionLightTest=(proba_predRF[:,1]+proba_predNN[:,1]+proba_predSVM[:,1]+proba_predSVM2[:,1]+proba_predKN[:,1])/float(5)
    y_pred_fusionLightTest = (y_pred_fusionLightTest > 0.5)
    
    f=open('FusionLight//ypredfusionLightTest.txt',"w") 
    np.save(f,y_pred_fusionLightTest)
    f.close()
    
    f=open('FusionLight//fusionLightTest.txt',"w") 
    np.save(f,fusionLightTest)
    f.close()
    
    f=open('FusionLight//fusionLightTargetTest.txt',"w") 
    np.save(f,fusionLightTargetTest)
    f.close()
    
    score = np.sum(y_pred_fusionLightTest==fusionLightTargetTest)/float(np.shape(y_pred_fusionLightTest)[0])
    
    f=open('FusionLight//scoreFusionLightTest.txt',"w") 
    f.write(str(score)+"\n")
    f.close()



  
def ATPpredict(test,targetTest,indiceBoolean):
    try:
        os.stat('ATP//')
    except:
        os.mkdir('ATP//')
    
    y_pred = np.array([])
    for i in range(np.shape(test)[0]):
        if (test[i][indiceBoolean]<0.5):
            y_pred=np.hstack((y_pred,np.array([1])))
        if (test[i][indiceBoolean]>0.5):
            y_pred=np.hstack((y_pred,np.array([0])))    
    
    f=open('ATP'+'//y'+'ATP'+'test.txt',"w") 
    np.save(f,y_pred)
    f.close()
    
    score=np.sum(y_pred==targetTest)/float(np.shape(targetTest)[0])
    f=open('ATP//scoreATPtest.txt',"w") 
    f.write(str(score)+"\n")
    f.close()

def createFolders(typeAlgo):
    #typeAlgo : RF, SVM, NN...
    try:
        os.stat(typeAlgo+'//')
    except:
        os.mkdir(typeAlgo+'//')
        
    try:
        os.stat(typeAlgo+'//clf')
    except:
        os.mkdir(typeAlgo+'//clf')
            
    
def main():
    RandomForest = 0
    NeuralNetwork = 0
    SupportVectorMachine = 0
    SupportVectorMachine2 = 0
    KN=1
    ATPb = 0
    FusionLight = 0
    Fusion = 0
    
    #Extraction/parsing
    try:
        data, labels, features, featuresReverse = dataParser.loadSavedData('Data')
        print 'Base loadée'
    except:
        data, labels, features, featuresReverse = dataParser.generateTestingData([0.4,0.4,0.1,0.1],2)
        dataParser.saveTreatedData('Data', data, labels, featuresReverse)
        print 'Base parsée'
        
    train = np.array(data[0])
    train2 = np.array(data[1])
    test = np.array(data[2])  
    test2 = np.array(data[3])
    
    targetTrain = np.array(labels[0])
    targetTrain2 = np.array(labels[1])
    targetTest = np.array(labels[2])  
    targetTest2 = np.array(labels[3])    
    
       
    f=open('test.txt',"w") 
    np.save(f,test)
    f.close()
  
    f=open('targetTest.txt',"w") 
    np.save(f,targetTest)
    f.close()
    
    f=open('test2.txt',"w") 
    np.save(f,test2)
    f.close()
  
    f=open('targetTest2.txt',"w") 
    np.save(f,targetTest2)
    f.close()
    
    f=open('train.txt',"w") 
    np.save(f,train)
    f.close()
  
    f=open('targetTrain.txt',"w") 
    np.save(f,targetTrain)
    f.close()
    
    f=open('train2.txt',"w") 
    np.save(f,train2)
    f.close()
  
    f=open('targetTrain2.txt',"w") 
    np.save(f,targetTrain2)
    f.close()
    if (RandomForest==1):  
        createFolders('RF')
        #Learning
        learn('RF',train,targetTrain,'learningRF.pk1')
        #Predicting    
        predict('RF',test,'learningRF.pk1','test')

    if (NeuralNetwork==1):  
        createFolders('NN')
        #Learning
        learn('NN',train,targetTrain,'learningNN.pk1')
        #Predicting    
        predict('NN',test,'learningNN.pk1','test')

        
    if (SupportVectorMachine==1):  
        createFolders('SVM')
        #Learning
        learn('SVM',train,targetTrain,'learningSVM.pk1')
        #Predicting
        predict('SVM',test,'learningSVM.pk1','test')

    if (KN==1):  
        createFolders('KN')
        #Learning
        learn('KN',train,targetTrain,'learningKN.pk1')
        #Predicting    
        predict('KN',test,'learningKN.pk1','test')

    if (SupportVectorMachine2==1):  
        createFolders('SVM2')
        #Learning
        learn('SVM2',train,targetTrain,'learningSVM2.pk1')
        #Predicting    
        predict('SVM2',test,'learningSVM2.pk1','test')

        
    if (ATPb==1):
        ATPpredict(test,targetTest,features['RankingBoolean'])
        
    if (FusionLight==1):           
        #Predicting    
        FusionLightpredict(train2,test2,targetTrain2,targetTest2)

        
    if (Fusion==1):           
        createFolders('Fusion')
        #Learning
        learn('Fusion',train2,targetTrain2,'learningFusion.pk1')
        #Predicting    
        predict('Fusion',test2,'learningFusion.pk1','test2')
main();
