# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:22:53 2021

@author: PC
"""
import pandas as pd
df = pd.read_csv('C:/Users/PC/OneDrive/Desktop/blogtext.csv')
df
df.isnull().any()
df.shape
data = df.head(10000)
data.info()
data.drop(['id','date'],axis=1)
data['age'] = data['age'].astype('object')
data.info()
pd.set_option('display.max_rows', None)
import nltk
import re
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
corpus = []
for i in range(len(data['text'])):
    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)
data['labels']=data.apply(lambda col: [col['gender'],str(col['age']),col['topic'],col['sign']], axis=1)
data['text'] = corpus
X = data['text']
Y = data['labels']

'''# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()'''

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X).toarray()
cv.get_feature_names()[:4322]
X[1]
label_counts = dict()
for labels in data.labels.values:
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
from sklearn.preprocessing import MultiLabelBinarizer
binarizer = MultiLabelBinarizer()
Y=binarizer.fit_transform(data.labels)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# RandomForestClassisfier
from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators=3000, criterion='gini')
Classifier.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
Y_pred = Classifier.predict(X_test)
print(Y_pred)
accuracy_RFC = accuracy_score(Y_test,Y_pred)
print(accuracy_RFC)
Y_pred_inversed = binarizer.inverse_transform(Y_pred)
Y_test_inversed = binarizer.inverse_transform(Y_test)
