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


def read_multiple_csv(file_path):
    # Use glob to match the pattern ‘csv’
    csv_files = glob.glob(file_path)

    # Create an empty list to hold dataframes
    list_data = []

    # Loop through the list of csv files
    for filename in csv_files:
        # Read each csv file and append it to the list_data
        data = pd.read_csv(filename)
        list_data.append(data)

    # Concatenate the dataframes in the list_data
    df = pd.concat(list_data, ignore_index=True)
    
    return df

def prepare_data(master_data: pd.DataFrame, top_words_count: int):
    master_data_preprocessed = master_data[["content", "account_category"]]
    master_data_preprocessed["troll"] = master_data_preprocessed["account_category"].apply(lambda x: 1 if (x == 'LeftTroll' or x == 'RightTroll') else 0)

    # Most used words in all data
    top_x_single_words = get_top_words(master_data_preprocessed, 0, top_words_count)
    
    for each_word in top_x_single_words.keys():
        master_data_preprocessed["Unigram: " + each_word] = master_data_preprocessed.content.str.count(" " + each_word + " ")
        print(each_word, "occurence", master_data_preprocessed["Unigram: " + each_word].sum(), "\n")

    # df['c2'] = df['c1'].apply(lambda x: 1 if x == 'LeftTroll' or x == 'RightTroll' else 0)

    return master_data_preprocessed

def get_top_words(dataset_with_text, m, n):
    top_words_origin = dataset_with_text["content"].str.replace(","," ").str.cat().split()
    top_words_base = {}
    for each_word in top_words_origin:
        if each_word in top_words_base: 
            top_words_base[each_word] += 1 
        else:
            top_words_base[each_word] = 1
    top_words_base = sorted(top_words_base.items(), key=operator.itemgetter(1))
    top_words_base = dict(top_words_base[::-1])
    
    top_words_base_top_n = {k: top_words_base[k] for k in list(top_words_base)[m:n]} 


    top_words_base_no_punc_words = {}
    for word, value in top_words_base_top_n.items():
        if word[-1] in string.punctuation: 
            word = word[:-1] 
        if len(word) > 0: 
            if word in top_words_base_no_punc_words: 
                top_words_base_no_punc_words[word] += value 
            else: 
                top_words_base_no_punc_words[word] = value
    top_words_base_no_punc_sorted = sorted(top_words_base_no_punc_words.items(), 
                                           key=operator.itemgetter(1)) 
    top_words_dict_final = dict(top_words_base_no_punc_sorted[::-1])
    return top_words_dict_final