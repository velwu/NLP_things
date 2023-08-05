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

def show_scores_and_plot_matrices(result_dicts):
    summed_matrix = np.array([[0, 0], [0, 0]])
    for each in result_dicts:
        df_cm = pd.DataFrame(each["Matrix"])
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        print("Fold", each["Fold Index"], "Accuracy:", each["Accuracy"], "Kappa:", each["Kappa"]) 
        summed_matrix = np.add(summed_matrix, each["Matrix"])  

    averaged_matrix = np.divide(summed_matrix, 4)
    
    plt.figure(figsize = (16,9))
    sns.set(font_scale=1.4)#label size
    sns.heatmap(averaged_matrix, cmap="Greens", annot=True,annot_kws={"size": 16})
    
    # Averaging the 4 folds might be better than adding all 4 together