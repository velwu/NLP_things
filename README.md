# Twitter Troll Dectection with N-gram, Decision Tree & Random Forest

A "project" which involves identifying Radical Left and Radical Right troll posts on Twitter using Decision Tree and Random Forest classifiers and 4-fold cross validation.

### Elements

These elements are involved in this project:

**Data Preprocessing**: Utilizing nltk (Natural Language Toolkit) to handle text preprocessing, including removing stop words and lemmatization, which is crucial in Natural Language Processing (NLP) tasks.

**Feature Extraction**: Imported `CountVectorizer` and `TfidfVectorizer` from `sklearn.feature_extraction.tex`t, which are important tools for converting text data into numerical vectors that can be used for machine learning.

**Modeling**: Included a variety of classifiers (`GaussianNB`, `MultinomialNB`, `KNeighborsClassifier`, `DummyClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`), which will allow trials and comparisons of different models along with their performances.

**Evaluation**: Functions like `train_test_split`,`KFold`, `cross_val_score`, and various metrics from sklearn will allow me to effectively evaluate the models.

The dataset used here is the [Russian Troll Tweets](https://www.kaggle.com/datasets/fivethirtyeight/russian-troll-tweets) data on Kaggle.

The dataset looks like this at a glance:
![IMG_DATA](https://github.com/velwu/NLP_things/blob/main/Dataset_Snapshot.PNG)

---

### Example for Model and Performance tuning

Back when i first did this project (year 2020), the most noticeable improvement observed was simply the increase of words-to-keep. When a set of about 50 features was increased ten-fold. Quite in the literal sense.

The process sees improvement from a original model built around minimalistic features (about 50 words to keep):
![IMG_PRIMITIVE_MDL](https://github.com/velwu/NLP_things/blob/main/Mdl_Perf_Before.PNG)

~~ to a better iteration using Decision Tree over 500 features of N-grams:
![IMG_BETTER_MDL](https://github.com/velwu/NLP_things/blob/main/Mdl_Perf_After.PNG)

However, this was the furthermost i was able to get. The additional points to consider are listed below. Hence i am rebooting this project to do what i envisioned to do.

---

### Additional points to consider

1. **Word Embeddings:** I am considering implementing word embeddings like Word2Vec or GloVe, as these techniques offer a more sophisticated representation of words compared to simple count-based methods. They are capable of capturing semantic meanings and could potentially improve the performance of my model.

2. **Dimensionality Reduction:** Depending on the size of my feature set after vectorization, I might need to utilize dimensionality reduction techniques like PCA (Principal Component Analysis) or SVD (Singular Value Decomposition). These techniques can be particularly beneficial if I opt to use word embeddings.

3. **Hyperparameter Tuning:** To optimize my models, I plan on using GridSearchCV or RandomizedSearchCV from `sklearn.model_selection` for hyperparameter tuning.

4. **Handling Imbalanced Classes:** If my classes turn out to be imbalanced (for example, more 'RightTroll' instances than 'LeftTroll'), I am prepared to use techniques for dealing with class imbalance. Libraries like SMOTE or ADASYN from `imblearn.over_sampling` may prove helpful in this context.

5. **Advanced NLP Techniques:** Depending on the complexity of my task, I might consider incorporating more advanced NLP techniques like sentiment analysis, POS tagging, named entity recognition, etc.

In my process, I will be starting with an exploratory data analysis and then proceed with handling any missing values, text data preprocessing, feature extraction, and finally, modeling. Visualizing the data and results at each step will be a crucial part of my workflow, helping me to understand the underlying patterns and workings of my model.
