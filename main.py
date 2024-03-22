import sklearn
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('~/Desktop/Projects/Multiple Disease Prediction/MultipleDiseaseDataset/Blood_samples_dataset_balanced_2(f).csv')
le = LabelEncoder()
sc = StandardScaler()

# Preprocessing Work
#print(df)
#df.info()
#print(df.isnull())
#print(df.isnull().sum())

df['Disease'] = le.fit_transform(df['Disease'])
X = df.iloc[:, :24]
y = df.iloc[:, 24:25]
y = le.fit_transform(y)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 1)
XtrainSc = sc.fit_transform(Xtrain)
XtestSc = sc.fit_transform(Xtest)


# K Nearest Neighbors
k = round(math.sqrt(len(ytrain)))
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(Xtrain, ytrain)
knn_pred = knn.predict(Xtest)
knn_accuracy = (metrics.accuracy_score(ytest, knn_pred)) * 100
print('K Nearest Neighbors Algorithm Accuracy Metric:',knn_accuracy,'%')


# K Nearest Neighbors - Scaled
knnSc = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
knnSc.fit(XtrainSc, ytrain)
knnSc_pred = knnSc.predict(XtestSc)
knnSc_accuracy = (metrics.accuracy_score(ytest, knnSc_pred)) * 100
print('K Nearest Neighbors Scaled Algorithm Accuracy Metric:',knnSc_accuracy,'%')