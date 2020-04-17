import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import keras
from keras.layers import Input, LSTM, Dense, GRU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model



data = np.array(pd.read_fwf("processed.cleveland.data"))#loads data
#############################################################################
#Module for preprocessing
def preprocess(data):
    temp_arr = []
    for rows in range(0, len(data)):
        if rows == 0:
            acc = data[rows,:][0];acc = acc.split(',')
            for item in range(0,len(acc)):
                 if acc[item] == '?':
                     temp_arr.append(new_data[rows-1,item])
                 else:
                     temp_arr.append(float(acc[item]))
            new_data = np.array(temp_arr).reshape(1,len(temp_arr) );temp_arr = []
        
        else:
            acc = data[rows,:][0];acc = acc.split(',')
            for item in range(0,len(acc)):
                if acc[item] == '?':
                    temp_arr.append(new_data[rows-1,item])
                else:
                    temp_arr.append(float(acc[item]))
    
            new_data = np.vstack(( new_data,  np.array(temp_arr).reshape(1,len(temp_arr) )  ))
            temp_arr = []
    return new_data
            
            
            
            
            
#######################################################################
def AB_train(Xtrain,ytrain):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(Xtrain, ytrain)
    return clf

def AB_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1  -    (np.count_nonzero(acc)  /   len(ytest))
    return acd, ba
            
def LR_train(Xtrain,ytrain):
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(Xtrain, ytrain)
    return clf
    
def LR_test(clf, Xtest,ytest):
    ba  = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1  -    (np.count_nonzero(acc)  /   len(ytest))
    return acd, ba
    
def SV_train(Xtrain,ytrain):
    clf = SVC(gamma='auto')
    clf = clf.fit(Xtrain, ytrain)
    return clf

def SV_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1  -    (np.count_nonzero(acc)  /   len(ytest))
    return acd, ba

    
    
def DNN_train(Xtrain,ytrain):
    num_class = 5
    num_features = 13
    n_epoch = 40
    n_batch = 10

    model = Sequential()
    model.add(Dense(256,input_shape=(num_features,)))
    model.add(Dense(256))
    model.add(Dense(num_class))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))



    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
    filepath="weights-improvement1-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath,monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    ytrain = np.array(pd.get_dummies(np.array(ytrain.astype(int).reshape(-1))))
    model.fit(Xtrain,ytrain, epochs=n_epoch, batch_size=n_batch, verbose=2)
    

    clf = model
    return clf
    
def DNN_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    ba = np.argmax(ba, axis=1) + 1
    acc = (ba-ytest);
    acd = 1  -    (np.count_nonzero(acc)  /   len(ytest))

    return acd, ba
 
 #######################################################################################
    

if __name__ == '__main__':
    data = preprocess(data)
    train_input = data[:,0:data.shape[1]-1]
    target_labels = data[:,data.shape[1]-1 ]
    Xtrain, Xtest, ytrain, ytest = train_test_split(train_input, target_labels, test_size=0.1, random_state=5)
    
    (unique, counts) = np.unique(ytrain, return_counts=True)
    print('train target distribution',(counts/len(ytrain)) )
    (unique, counts) = np.unique(ytest, return_counts=True)
    print('test target distribution',(counts/len(ytest)) )



    #Models , Training and testing
    #Adaboost
    clf1 = AB_train(Xtrain,ytrain)
    acc, ba = AB_test(clf1, Xtest,ytest)
    print('adaboost accuracy is ', acc)
    precision, recall, fscore, support = score(ytest, ba)
    print('fscore: {}'.format(fscore))
    
    #Logistic regression
    clf2 = LR_train(Xtrain,ytrain)
    acc, ba  = LR_test(clf2, Xtest,ytest)
    print('logistic regression accuracy is ', acc)
    precision, recall, fscore, support = score(ytest, ba)
    print('fscore: {}'.format(fscore))
    
    
    #Support vector
    clf3 = SV_train(Xtrain,ytrain)
    acc, ba  =  SV_test(clf3, Xtest,ytest)
    print('support vector machine accuracy is ', acc)
    precision, recall, fscore, support = score(ytest, ba)
    print('fscore: {}'.format(fscore))
    
    #Deep learning
    clf4 = DNN_train(Xtrain,ytrain)
    clf4 = load_model('stroke')
    acc, ba = DNN_test(clf4, Xtest,ytest)
    print('DNN accuracy is ',  acc)
    precision, recall, fscore, support = score(ytest, ba)
    print('fscore: {}'.format(fscore))
  
    
    

