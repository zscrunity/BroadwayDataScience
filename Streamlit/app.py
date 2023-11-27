import streamlit as st
import numpy as np
import pandas as pd
import re,string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.ensemble import RandomForestClassifier
import pickle
import nltk
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
st.title('Hamburger')

df = pd.read_csv('Cleaned_BBC_text.csv', encoding = 'latin1')
df = df.sample(frac = 1)

vectorizer = TfidfVectorizer(stop_words="english")

X = df['cleaned']
Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1450)),
                     ('clf', LogisticRegression(random_state=0))])

model = pipeline.fit(X_train, y_train)

file = open('news.txt' , 'r')
news = file.read()
file.close()

news_data = {'predict_news':[news]}
news_data_df = pd.DataFrame(news_data)

predict_news_cat = model.predict(news_data_df['predict_news'])
st.write("Predicted news category = ",predict_news_cat[0])
st.write(news)
