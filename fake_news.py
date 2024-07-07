import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

news_dataset = pd.read_csv(r"C:\Users\HP\Downloads\train.csv\train.csv")

news_dataset.shape

news_dataset.head()

news_dataset.isnull().sum()

news_dataset = news_dataset.fillna('')

data_true_manual_testing.head(10)

news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

print(news_dataset['content'])

X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)
print(Y)

Y.shape

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

print(Y_test[3])

from sklearn.ensemble import RandomForestClassifier

Ran=RandomForestClassifier()
Ran.fit(X_train,Y_train)

ran_train_pred=Ran.predict(X_train)
training_data_accuracy1 = accuracy_score(ran_train_pred, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy1)

ran_test_pred = model.predict(X_test)
test_data_accuracy2 = accuracy_score(ran_test_pred, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy2)

from sklearn.tree import DecisionTreeClassifier

dec=DecisionTreeClassifier()
dec.fit(X_train,Y_train)

dec_train_pred=dec.predict(X_train)
training_data_accuracy_dec = accuracy_score(dec_train_pred, Y_train)

print('Accuracy score of the test data : ', training_data_accuracy_dec)

dec_test_pred=dec.predict(X_test)
test_data_accuracy_dec = accuracy_score(dec_test_pred, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy_dec)

from sklearn.ensemble import GradientBoostingClassifier

gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, Y_train)

y_train_pred = gb_classifier.predict(X_train)

accuracy = accuracy_score(Y_train, y_train_pred)

print(f"Accuracy: {accuracy}")

y_test_pred = gb_classifier.predict(X_test)

accuracy = accuracy_score(Y_test, y_test_pred)

print(f"Accuracy: {accuracy}")
