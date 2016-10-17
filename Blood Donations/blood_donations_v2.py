# -*- coding: utf-8 -*-
"""
@author: igor

Implementing a cross-validated classifier solution. 

Included models are:  
* Logistic regression  
* Linear discriminant analysis  
* K neighbors classifier  
* Decision tree classifier  
* Naive Bayes  
* SVMclassifier  

KNN performs the best with ~0.72 precision and ~0.45 recall on predicting donations
"""
# Load libraries
import os
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Set up data frames
dir = "/Users/igor/Small Projects/Blood Donations"
r_train = "9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv"

r_train_names = ['id', 'm_since_donation', 'donation_count',  'donation_volume', 'm_since_first_donation', 'donation_flag']

train = pd.read_csv(os.path.join(dir,r_train), names = r_train_names, skiprows= [0])

# CONVERTING DataFrame into Numpy arrays.
# This data is clean enough that is not necessary, but a good precaution 
# when working ith SciKit Learn

X_train = train.drop(['id', 'donation_flag'], 1).as_matrix()
Y_train = train['donation_flag'].as_matrix()


## EXPLORE DATA
# What is the data structure? 
train.shape
train.dtypes
train.describe()

# VISUALIZE DATA
# scatter plot matrix
scatter_matrix(train)
plt.show()

# Multiple records per ID? FALSE
len(train['id']) != len(set(train['id']))
train.std()

# Compare groups. 
grouped = train.groupby('donation_flag')
grouped.describe()
grouped.hist()
grouped.boxplot()



# MODEL BUILDING 
# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
 
 
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show() 


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_train)
print(accuracy_score(Y_train, predictions))
print(confusion_matrix(Y_train, predictions))
print(classification_report(Y_train, predictions))