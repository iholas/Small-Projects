# Blood Donations 
Idea from: [Driven Data](https://www.drivendata.org/competitions/2/)

## Original version (October 10th)
Implementing three classification methods to predict blood donation
1) simple lgistic regression
2) Stochaistic Gradient Descent Classifier
3) ADA Boosted Decision Tree Classifier 

Currently the SGDC method is very lousy, must try to improve.  
Logistic regression has decent accuracy (~0.76), but poor recall (0.1).   
ADABoost Classifier has good Accuracy (~0.81), but still sub 0.5 recall. 

## Version 2 (October 17th)
Implementing a cross-validated classifier solution. 

Included models are:  
* Logistic regression  
* Linear discriminant analysis  
* K neighbors classifier  
* Decision tree classifier  
* Naive Bayes  
* SVMclassifier  

KNN performs the best with ~0.72 precision and ~0.45 recall on predicting donations


