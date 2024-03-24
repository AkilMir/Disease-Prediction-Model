import sklearn
import warnings
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

train_df = pd.read_csv('~/Desktop/Projects/Disease Prediction Model/MultipleDiseaseDataset/Blood_samples_dataset_balanced_2(f).csv')
test_df = pd.read_csv('~/Desktop/Projects/Disease Prediction Model/MultipleDiseaseDataset/blood_samples_dataset_test.csv')
le = LabelEncoder()
sc = StandardScaler()

df = pd.concat([train_df, test_df], ignore_index=True)
df['Disease'] = le.fit_transform(df['Disease'])

# Preprocessing Work
print(df)
df.info()
print(df.isnull())
print(df.isnull().sum())


X = df.iloc[:, :24]
y = df.iloc[:, 24:25]
y = le.fit_transform(y)



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 1)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)


# K Nearest Neighbors
k = round(math.sqrt(len(ytrain)))
# Through trial and error, a k value of 2 is yielding a higher accuracy score than the convention used for calculating k (prev. line)
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(Xtrain, ytrain)
knn_pred = knn.predict(Xtest)
knn_accuracy = (metrics.accuracy_score(ytest, knn_pred)) * 100
print('K Nearest Neighbors Algorithm Accuracy Metric:',knn_accuracy,'%')


#sns.pairplot(data = df, hue = 'Disease', palette = 'Set2')
#plt.show() 

# Support Vector Machine
# Rbf kernel yields higher accuracy score than the linear kernel
svm=SVC(kernel="rbf", random_state= 0)
svm.fit(Xtrain, ytrain)
svm_pred = svm.predict(Xtest)
svm_accuracy = (metrics.accuracy_score(ytest, svm_pred)) * 100
print('Support Vector Machine Algorithm Accuracy Metric:',svm_accuracy,'%')


# Logistic Regression
lreg = LogisticRegression()
lreg.fit(Xtrain, ytrain)
lreg_pred = lreg.predict(Xtest)
lreg_accuracy = (metrics.accuracy_score(ytest, lreg_pred)) * 100
print('Logistic Regression Algorithm Accuracy Metric:',lreg_accuracy,'%')
