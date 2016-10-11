# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:11:11 2016

@author: igor

Implementing three classification methods to predict blood donation
1) simple lgistic regression
2) Stochaistic Gradient Descent Classifier
3) ADA Boosted Decision Tree Classifier 

Currently the SGDC method is very lousy, must try to improve
Logistic regression has decent accuracy. but poor recall
ADABoost Classifier has good Accuracy, but still sub 0.5 recall.  
"""

import pandas as pd
import os
from sklearn import linear_model
from sklearn import ensemble
# import seaborn as sns
# import numpy as np
from ggplot import ggplot, aes, geom_point

# FUNCTION FOR EVALUATING AND PLOTTING THE RESULTS
def eval(df_in, predicted, method):
    print(method)
    
    df = df_in
    
    df['Correct']= df[predicted] == df['donation_flag']
    df['Class'] = 'True Positive'
    df['Class'][(df[predicted] == 1) & (df['Correct'] == False)] = 'False Positive'
    df['Class'][(df[predicted] == 0) & (df['Correct'] == True)] = 'True Negative'
    df['Class'][(df[predicted] == 0) & (df['Correct'] == False)] = 'False Negative'
    
    TP = df[(df['Class'] == 'True Positive')].shape[0]
    FP = df[(df['Class'] == 'False Positive')].shape[0]
    TN = df[(df['Class'] == 'True Negative')].shape[0]
    FN = df[(df['Class'] == 'False Negative')].shape[0]
    
    print ggplot(df, aes(x='donation_count', y='m_since_donation', color = 'Class')) + geom_point()
    
    confusion = pd.DataFrame({'Positive': [FP, TP],
                              'Negative': [TN, FN]},
                              index = ['TrueNeg', 'TruePos'])
    accuracy = float(TP+TN)/float(TP + TN + FP + FN)
    precision = float(TP)/float(TP + FP)
    recall = float(TP)/float(TP + FN)
    
    print(confusion)
    print('accuracy = ' + str(accuracy))
    print('precision = ' + str(precision))
    print('recall = ' + str(recall))
    print('Done')

# Set up data frames
dir = "/Users/igor/Small Projects/Blood Donations"
r_train = "9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv"

r_train_names = ['id', 'm_since_donation', 'donation_count',  'donation_volume', 'm_since_first_donation', 'donation_flag']

train = pd.read_csv(os.path.join(dir,r_train), names = r_train_names, skiprows= [0])

# CONVERTING DataFrame into Numpy arrays.
# This data is clean enough that is not necessary, but a good precaution 
# when working ith SciKit Learn

x = train.drop(['id', 'donation_flag'], 1).as_matrix()
y = train['donation_flag'].as_matrix()


## EXPLORE DATA
# What is the data structure? 
train.dtypes
train.describe()

# Multiple records per ID? FALSE
len(train['id']) != len(set(train['id']))
train.std()

# Compare groups. 
grouped = train.groupby('donation_flag')
grouped.describe()
grouped.hist()
grouped.boxplot()


# Logistic Model
logit = linear_model.LogisticRegression()
logit.fit(x,y)
logit.get_params()
train['y_logit_cat'] = logit.predict(x)
train['y_logit_prob'] = logit.predict_proba(x)[:,1]
eval(train, 'logit_predicted', 'Logistic Regression')

# SGDC
# SGDC
SGDC = linear_model.SGDClassifier(shuffle = True)
SGDC.fit(x,y)
SGDC.get_params()
train['y_SGDC_df'] = SGDC.decision_function(x)
train['y_SGDC_cat'] = SGDC.predict(x)
eval(train, 'y_SGDC_cat', 'SGDC')

# ADA Boost
ada = ensemble.AdaBoostClassifier()
ada.fit(x,train['donation_flag'])
ada.get_params()
train['y_ada_cat'] = ada.predict(x)
train['y_ada_coeff'] = ada.predict_proba(x)[:,1]
eval(train, 'y_ada_cat', 'ADABoost Claffier')

