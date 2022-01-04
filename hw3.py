# CSES Population Survey Voting Prediction - CSSM 502 Homework 3 

#Importing the Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Reading the Data 
df = pd.read_csv("cses4_cut.csv") #reading the data
df.drop(["Unnamed: 0"], axis=1, inplace=True) #unnecessary index-like column 
df.head()
df["voted"].value_counts() #voted value counts
df.corr().loc['voted'] #correlation of every feature with the target
df.columns #columns 

# Printing the value counts for every column to see which variables are the best predictors, which have the least missing values. 
for c in df.columns: 
    print ("---- %s ---" % c)
    print (df[c].value_counts()) 


# 1. D2002, 9=0 (Excellent percentage)
# 
# 2. D2003, 99=18 (Good percentage)
# 
# 3. D2004 9=738 (10% missing) 
# 
# 4. D2005 9=1108 (12% missing) 
#  
# 5. D2006 9=2206 (15% missing) 
# 
# 5. D2007 9=7086 (Majority missing) 
# 
# 6. D2008 9=7087 (Majority missing) 
# 
# 7. D2009 9=7086 (Majority missing)
# 
# 8. D2010 99=3 (Excellent) 
# 
# 9. D2011 999=7566 (Majority missing) 
# 
# 10. D2012 9=6556 (Majority missing) 
# 
# 11. D2013 9=5227 (Majority missing) 
# 
# 12. D2014 9=5398 (Almost half of it missing) 
# 
# 13. D2015 99=7063 (Majority missing) 
# 
# 14. D2016 999= 10444(Huge majority missing) 
# 
# 15. D2017 9=10481 (Huge majority missing) 
# 
# 16. D2018 9=9214 (Huge majority missing) 
# 
# 17. D2019 9=9132 (Huge majority missing)
# 
# 18. D2020 9=382 (Good percentage)
# 
# 19. D2021 99=1887 (Not bad)
# 
# 20. D2022 99=3417 (Almost half of it missing) 
# 
# 21. D2023 99=4048 (Almost half of it missing)
# 
# 22. D2024 9=239 (Really good) 
# 
# 23. D2025 9=2539 (Half of it missing) 
# 
# 24. D2026 9=0 (Excellent)
# 
# 25. D2027 999=4823 (Majority missing) 
# 
# 26. D2028 99=1103 (Almost two thirds missing) 
# 
# 27. D2029 999=5539 (Almost half of it missing) 
# 
# 28. D2030 999=10072 (Huge majority missing) 
# 
# 29. D2031 9=4822 (Almost half of it missing) 


# By looking at these approximations, intuitively, we can see that;

# 1.D2002 (Sex), 
# 
# 2.D2003 (Education),
# 
# 3.D2010 (Current Employment Status),
# 
# 4.D2020 (Household Income),
# 
# 5.D2021(Number in Household Total),
# 
# 6.D2024(Religious Services Attendance),
#
#are the features with the least missing values and make theoratical sense which might suggest that we might build our model on them. 

X = df[['D2002', 'D2003', 'D2010', 'D2020', 'D2021','D2024','age']]
corr_X = df[['D2002', 'D2003', 'D2010', 'D2020', 'D2021','D2024','age','voted']]
y = df["voted"]

#Imputing the missing values with the most frequent values in the column. 

imp_freq_9 = SimpleImputer(missing_values=9, strategy="most_frequent")
imp_freq_99 = SimpleImputer(missing_values=99, strategy="most_frequent")

d2003imp = imp_freq_99.fit_transform(X["D2003"].to_numpy().reshape(-1,1))
d2024imp = imp_freq_9.fit_transform(X["D2024"].to_numpy().reshape(-1,1))
d2010imp = imp_freq_99.fit_transform(X["D2010"].to_numpy().reshape(-1,1))
d2021imp = imp_freq_99.fit_transform(X["D2021"].to_numpy().reshape(-1,1))
d2020imp = imp_freq_9.fit_transform(X["D2020"].to_numpy().reshape(-1,1));

X["D2024"] = d2024imp
X["D2003"] = d2003imp
X["D2010"] = d2010imp
X["D2020"] = d2020imp
X["D2021"] = d2021imp
corr_X["D2024"] = d2024imp
corr_X["D2003"] = d2003imp
corr_X["D2010"] = d2010imp
corr_X["D2020"] = d2020imp
corr_X["D2021"] = d2021imp

X["D2024"].value_counts() #checking...notice there are no 9's (the missing value)
corr_X.corr().loc['voted'] #the correlations of the features to the voting behavior. 


#One-Hot Coding

voted = pd.get_dummies(y,prefix="Voted",drop_first=True)
xdum = pd.get_dummies(data=X, columns= ['D2002', 'D2003', 'D2010', 'D2020', 'D2021','D2024'],drop_first=True)
data =  pd.concat([xdum,voted], axis=1)

#Train-test split 

X = data.iloc[:,:-1]
y = data["Voted_True"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=1)


#Classifier Algorithms

#Trying out many algorithms to find the best fit. 

#GaussianNB Classifier
model = GaussianNB()
model.fit(Xtrain, ytrain) 
predictGNB = model.predict(Xtest)
accuracy_score(ytest,predictGNB) 

print(confusion_matrix(ytest,predictGNB)) #printing out the confusion matrix and the classification report for Gaussian Naive Bayes 
print("\n")
print(classification_report(ytest,predictGNB))

np.set_printoptions(suppress=True) 
matNB = confusion_matrix(ytest, predictGNB) 
sns.heatmap(matNB, square=True, annot=True, cbar=False,fmt="g"); #Visualizing the GNB Confusion Matrix
plt.xlabel('Predicted value');
plt.ylabel('True value');

from sklearn.model_selection import cross_val_score #GNB cross-validation score 
np.mean(cross_val_score(model, X, y, cv=5))  

#Leave One Out 

from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut()) #loo
scores
scores.mean()

#Logistic Regression 

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(Xtrain,ytrain)
predictLR = lr.predict(Xtest)
accuracy_score(ytest,predictLR) 

print(confusion_matrix(ytest,predictLR))
print("\n")
print(classification_report(ytest,predictLR)) #Logistic Regression confusion matrix and classification report

np.set_printoptions(suppress=True) 
matLR = confusion_matrix(ytest, predictLR)
sns.heatmap(matLR, square=True, annot=True, cbar=False,fmt="g"); #visualizing the Logistic Regression confusion matrix 
plt.xlabel('Predicted value');
plt.ylabel('True value');

from sklearn.model_selection import cross_val_score 
np.mean(cross_val_score(lr, X, y, cv=5))  #Logistic Regression cross-validation score  

#KNeighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(Xtrain, ytrain) 
predictKN = model.predict(Xtest)
accuracy_score(ytest, predictKN)

np.set_printoptions(suppress=True)  
matKN = confusion_matrix(ytest, predictKN) #visualizing the KNeighbors confusion matrix
sns.heatmap(matKN, square=True, annot=True, cbar=False,fmt="g");
plt.xlabel('Predicted value');
plt.ylabel('True value');

print(confusion_matrix(ytest,predictKN))
print("\n")
print(classification_report(ytest,predictKN)) #KNeighbors confusion matrix and classification report 

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(Xtrain, ytrain)
predictRF = model.predict(Xtest)
accuracy_score(ytest,predictRF)
np.mean(cross_val_score(model, X, y, cv=5))

print(confusion_matrix(ytest,predictRF))
print("\n")
print(classification_report(ytest,predictRF)) #Random Forest confusion matrix and classification report

#Support Vector Machine
from sklearn.svm import SVC
model = SVC()
model.fit(Xtrain,ytrain)
predictSVC = model.predict(Xtest)
accuracy_score(ytest,predictSVC)

print(confusion_matrix(ytest,predictSVC))
print("\n")
print(classification_report(ytest,predictSVC)) #Seems really problematic in predicting the false voting results. 

#Model Tuning 

#Here we can see that our models are predicting 1s disproportionately. The reason for that might be because the model parameters need adjusting or the model doesn't have enough False voting data to learn to classify them correctly.  We will be doing the tuning via the GridSearchCV.

#Grid Search 

#Tuning the SVC model
from sklearn.model_selection import GridSearchCV

SVC #tab + shift to explore the model parameters. 

"""SVC(
    *,
    C=1.0, #Large C value gives you low bias but high variance.You penalize the cost of missclassification with a large C value. Smaller C value gives you high bias and low variance. 
    kernel='rbf',#Default
    degree=3,
    gamma='scale', #A small gamma high bias and low variance in the model. 
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape='ovr',
    break_ties=False,
    random_state=None,
)"""

param_grid = {"C":[0.1,1,10,100,1000], "gamma":[1,0.1,0.01,0.001,0.0001],'kernel': ['rbf']} 
#This is the grid I will feed to the GridSearchCV which contains the parameters to tune 
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(Xtrain,ytrain)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(Xtest)

print(confusion_matrix(ytest,grid_predictions)) #tuning the SVC algorithm generates much better results, but still, the model needs to be improved or data sample size should be increased. 
print("\n")
print(classification_report(ytest,grid_predictions))


# Please check out pdf for more detailed information.
