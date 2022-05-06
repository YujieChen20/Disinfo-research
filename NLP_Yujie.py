# Read data

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report # accuracy_score
from pprint import pprint
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import random
from xgboost import XGBClassifier

df = pd.read_csv('NLP.csv')
df = df[:100000]

# Clean Tweets
stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from", "rt"]


def clean_tweet(tweet):
    if type(tweet) == float:
        return ""
    porter = PorterStemmer()
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    #temp = re.sub(r'\d+', '', temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp


Tweets_List = [clean_tweet(tw) for tw in df['status_text']]
Labels_List = [lab for lab in df['sentiment_label']]

# Instantiate CV first
CV_content = TfidfVectorizer(input='content',
                        stop_words='english',
                        #max_features=100
                        )
DTM = CV_content.fit_transform(Tweets_List)
ColNames = CV_content.get_feature_names_out()
DF_content = pd.DataFrame(DTM.toarray(), columns=ColNames)
DF_content.head()

X_train, X_test, y_train, y_test = train_test_split(DF_content, Labels_List, test_size=0.2, random_state=99)


def logisticRegr(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # save the model to disk
    filename_lr = 'logisticRegr.sav'
    pickle.dump(model, open(filename_lr, 'wb'))
    return model


def naiveBayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    # save the model to disk
    filename_nb = 'NaiveBayes.sav'
    pickle.dump(model, open(filename_nb, 'wb'))
    return model


def SVM(X_train, y_train):
    model = svm.SVC()
    model.fit(X_train, y_train)
    # save the model to disk
    filename_nb = 'SVM.pkl'
    pickle.dump(model, open(filename_nb, 'wb'))
    return model


def xgbc(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # save the model to disk
    filename_xgbc = 'xgbc.pkl'
    pickle.dump(model, open(filename_xgbc, 'wb'))
    return model


# load the model from disk
def loadModel(filename, X_test, y_test):
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred_test = loaded_model.predict(X_test)
    scores_test = classification_report(y_test, y_pred_test, output_dict=True)
    print(scores_test["accuracy"])
    cm = metrics.confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Accuracy Score:' + str(scores_test["accuracy"]), fontsize=16)
    plt.suptitle(str(filename.split(".")[0]) + 'Model', fontsize=24, y=1)
    plt.savefig(str(filename.split(".")[0]))
    return loaded_model


#y_pred_train = logisticRegr.predict(X_train)
#scores_train = classification_report(y_train, y_pred_train, output_dict=True)
#print("Training accuracy of Logistic Regression model:", scores_train["accuracy"])
#pprint(scores_train)

#y_pred_test = logisticRegr.predict(X_test)
#scores_test = classification_report(y_test, y_pred_test, output_dict=True)
#print("Testing accuracy of Logistic Regression model:", scores_test["accuracy"])
#pprint(scores_test)


# Cross Validation
def crossValidation(X, y, model, k=10):
    cv = RepeatedStratifiedKFold(n_splits=k, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print("Cross Validation Average Score", np.mean(scores))


X = DF_content
y = Labels_List

if __name__ == '__main__':
    #print('Fit Logistic Regression Model...')
    #lr = logisticRegr(X_train, y_train)
    #print('Fit Naive Bayes Model...')
    #nb = naiveBayes(X_train, y_train)
    #print('Load Logistic Regression Model...')
    #lr = loadModel('logisticRegr.sav', X_test, y_test)
    #print('Load Naive Bayes Model...')
    #nb = loadModel('NaiveBayes.sav', X_test, y_test)
    #print('5-fold cross validation result of Logistic Regression Model: ')
    #crossValidation(X, y, lr, 5)
    #print('5-fold cross validation result of Naive Bayes Model: ')
    #crossValidation(X, y, nb, 5)
    #print('Fit SVM Model...')
    #svm = SVM(X_train, y_train)
    xgbc = xgbc(X_train, y_train)
    #print('Load SVM Model...')
    #svm = loadModel('SVM.pkl', X_test, y_test)
    #print('5-fold cross validation result of SVM Model: ')
    #crossValidation(X, y, svm, 5)
