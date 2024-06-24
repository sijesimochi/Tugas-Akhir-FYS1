import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import joblib
import platform
import os
import time

# Read csv and check the data
cwd = os.path.dirname(os.path.abspath('_file_'))
dataset_dir = os.path.join(cwd, 'Transformed dataset')
mean_path = os.path.join(dataset_dir, 'transformed_mean.csv')
var_path = os.path.join(dataset_dir, 'transformed_var.csv')

mean_csv = pd.read_csv(mean_path)
var_csv = pd.read_csv(var_path)

mean_csv = mean_csv.drop(columns=["label"])
mean_csv = mean_csv.add_suffix("_mean")
mean_csv.columns = mean_csv.columns.str.replace("Unnamed: 0_mean", "index")

var_csv = var_csv.add_suffix("_var")
var_csv.columns = var_csv.columns.str.replace("Unnamed: 0_var", 'index')
var_csv.columns = var_csv.columns.str.replace("label_var", 'label')

merged = mean_csv.merge(var_csv, on='index')

features = merged.drop(['label', 'index'], axis=1)
X = features.values
Y = merged['label'].values

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.10, random_state=1)

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
# dt_predictions = dt.predict(X_val)

nb = GaussianNB()
nb.fit(X_train, Y_train)
# nb_predictions = nb.predict(X_val)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
# knn_predictions = knn.predict(X_val)

svm = SVC()
svm.fit(X_train, Y_train)
# svm_predictions = svm.predict(X_val)


joblib.dump(dt, 'Model/decisionTree.pkl')
joblib.dump(knn, 'Model/knn.pkl')
joblib.dump(nb, 'Model/naivebayes.pkl')
joblib.dump(svm, 'Model/svm.pkl')