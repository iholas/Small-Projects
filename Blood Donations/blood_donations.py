# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:11:11 2016

@author: igor

Implementing three classification methods to predict blood donation
1) simple lgistic regression
2) Stochaistic Gradient Descent Classifier
3) ADA Boosted Decision Tree Classifier 

Included is a comparison of the results. 
ADA Boost > Logistic > SGDC
"""

import pandas as pd
import os
from sklearn import linear_model
from sklearn import ensemble

# Set up data frames
dir = "/Users/____/Blood Donations"
r_train = "9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv"
r_test = "5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv"

r_train_names = ['id', 'm_since_donation', 'donation_count',  'donation_volume', 'm_since_first_donation', 'donation_flag']

train = pd.read_csv(os.path.join(dir,r_train), names = r_train_names, skiprows= [0])
test = pd.read_csv(os.path.join(dir,r_test), names = r_train_names[:5], skiprows= [0])

train_x = train.drop(['id', 'donation_flag'], 1)
test_x = train.drop(['id'], 1)

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
logit.fit(train_x,train['donation_flag'])
logit.get_params()
y_logit_df = logit.decision_function(train_x)
y_logit_cat = logit.predict(train_x)
sum(y_logit_cat == train['donation_flag'])/576.
y_logit_prob_test = logit.predict_proba(test_x)

# SGDC
SGDC = linear_model.SGDClassifier(shuffle = True)
SGDC.fit(train_x,train['donation_flag'])
SGDC.get_params()
y_SGDC_df = SGDC.decision_function(train_x)
y_SGDC_cat = SGDC.predict(train_x)
sum(y_SGDC_cat == train['donation_flag'])/576.

# ADA Boost
ada = ensemble.AdaBoostClassifier()
ada.fit(train_x,train['donation_flag'])
ada.get_params()
y_ada_cat = ada.predict(train_x)
y_ada_coeff = ada.predict_proba(train_x)
sum(y_ada_cat == train['donation_flag'])/576.
