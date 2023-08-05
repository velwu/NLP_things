import sklearn
import pandas as pd
import glob
import numpy as np
import scipy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly
import xlrd
import xml.etree.ElementTree as etree
import xmltodict
# packages that I think is especially useful for this topic~~
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import string
import itertools
import operator
import re
import copy
import unicodedata
import pickle

from nltk.corpus import stopwords
from scipy import stats
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

def predict_using_MultinomialNB(tweets_dataset):
    fold_result_dicts = []
    
    X = tweets_dataset[tweets_dataset.columns.difference(['content', 'account_category', 'troll'])]
    # X = tweets_dataset[tweets_dataset.columns.difference(['troll'])]
    y = tweets_dataset['troll']
    kf = KFold(n_splits=4, random_state=577, shuffle=True)

    fold_idx = 1
    print("MultinomialNB with unigrams as features.",)
    for train_index, test_index in kf.split(X):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            multiNB_mdl = MultinomialNB()        
            multiNB_mdl.fit(X_train, y_train)

            y_pred = multiNB_mdl.predict(X_test)

            accuracy_score_lr = metrics.accuracy_score(y_test, y_pred)
            kappa_lr = cohen_kappa_score(y_test, y_pred)
            confustion_matrix_lr = metrics.confusion_matrix(y_test, y_pred)
            
            single_fold_dict = {
                "Fold Index": fold_idx,
                "Accuracy": accuracy_score_lr,
                "Kappa": kappa_lr,
                "Matrix": np.array(confustion_matrix_lr)
            }
            
            fold_result_dicts.append(single_fold_dict)
            
            print("Fold No.", fold_idx, 
                  ", Accuracy:", accuracy_score_lr,
                  ", Kappa:", kappa_lr)
            print(confustion_matrix_lr)
            fold_idx += 1
    return fold_result_dicts

def predict_using_DecisionTree(tweets_dataset):
    fold_result_dicts = []
    
    X = tweets_dataset[tweets_dataset.columns.difference(['content', 'account_category', 'troll'])]
    # X = tweets_dataset[tweets_dataset.columns.difference(['troll'])]
    y = tweets_dataset['troll']
    kf = KFold(n_splits=4, random_state=577, shuffle=True)

    fold_idx = 1
    print("Decision Tree method that can be recycled. Sounds eco-friendly amIRight",)
    for train_index, test_index in kf.split(X):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            dtc_mdl = DecisionTreeClassifier()        
            dtc_mdl.fit(X_train, y_train)

            y_pred = dtc_mdl.predict(X_test)

            accuracy_score_lr = metrics.accuracy_score(y_test, y_pred)
            kappa_lr = cohen_kappa_score(y_test, y_pred)
            confustion_matrix_lr = metrics.confusion_matrix(y_test, y_pred)
            
            single_fold_dict = {
                "Fold Index": fold_idx,
                "Accuracy": accuracy_score_lr,
                "Kappa": kappa_lr,
                "Matrix": np.array(confustion_matrix_lr)
            }
            
            fold_result_dicts.append(single_fold_dict)
            
            print("Fold No.", fold_idx, 
                  ", Accuracy:", accuracy_score_lr,
                  ", Kappa:", kappa_lr)
            print(confustion_matrix_lr)
            fold_idx += 1
    return fold_result_dicts

def predict_using_RandomForest(tweets_dataset):
    fold_result_dicts = []
    
    X = tweets_dataset[tweets_dataset.columns.difference(['content', 'account_category', 'troll'])]
    # X = tweets_dataset[tweets_dataset.columns.difference(['troll'])]
    y = tweets_dataset['troll']
    kf = KFold(n_splits=4, random_state=577, shuffle=True)

    fold_idx = 1
    print("A forest of so many decision trees. Sounds quadruple eco-friendly amIRight",)
    for train_index, test_index in kf.split(X):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)

            y_pred=clf.predict(X_test)

            accuracy_score_lr = metrics.accuracy_score(y_test, y_pred)
            kappa_lr = cohen_kappa_score(y_test, y_pred)
            confustion_matrix_lr = metrics.confusion_matrix(y_test, y_pred)
            
            single_fold_dict = {
                "Fold Index": fold_idx,
                "Accuracy": accuracy_score_lr,
                "Kappa": kappa_lr,
                "Matrix": np.array(confustion_matrix_lr)
            }
            
            fold_result_dicts.append(single_fold_dict)
            
            print("Fold No.", fold_idx, 
                  ", Accuracy:", accuracy_score_lr,
                  ", Kappa:", kappa_lr)
            print(confustion_matrix_lr)
            fold_idx += 1
    return fold_result_dicts
