# Twitter Troll Dectection with N-gram, Decision Tree & Random Forest

A "project" which involves identifying Radical Left and Radical Right troll posts on Twitter using Decision Tree and Random Forest classifiers and 4-fold cross validation. Both Jupyter Notebook and HTML versions are included for the reader to inspect whichever suits his/her convenience.

The dataset used here is the [Russian Troll Tweets](https://www.kaggle.com/datasets/fivethirtyeight/russian-troll-tweets) data on Kaggle.

The dataset looks like this at a glance:
![IMG_DATA](https://github.com/velwu/NLP_things/blob/main/Dataset_Snapshot.PNG)

The process sees improvement from a original model built around minimalistic features (about 50 words to keep):
![IMG_PRIMITIVE_MDL](https://github.com/velwu/NLP_things/blob/main/Mdl_Perf_Before.PNG)

~~ to a better iteration using Decision Tree over 500 features of N-grams:
![IMG_BETTER_MDL](https://github.com/velwu/NLP_things/blob/main/Mdl_Perf_After.PNG)
