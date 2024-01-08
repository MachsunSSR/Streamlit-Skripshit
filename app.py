import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import os

# Get current working directory
cwd = os.getcwd()
st.title('Rule-Based Chatbot HaloFilkom')

data_path = os.path.join(cwd, 'data_merge_cleaned_low_with_response.csv')
df = pd.read_csv(data_path)

# Import model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

import xgboost as xgb

# Load TF-IDF in current path
tfidf_path = os.path.join(cwd, 'tfidf.sav')
tfidf = pickle.load(open(tfidf_path, 'rb'))

# Load Model XGB in current path
xgb_path = os.path.join(cwd, 'xgb_tfidf.sav')
xgb_tfidf = pickle.load(open(xgb_path, 'rb'))

# import label encoder
le_path = os.path.join(cwd, 'label_encoder.sav')
le = pickle.load(open(le_path, 'rb'))


def predict(kalimat):
    # Test model 

    #lower text
    kalimat = [x.lower() for x in kalimat]
    # Remove punctuation
    kalimat = [re.sub('[%s]' % re.escape(string.punctuation), '', x) for x in kalimat]
    # Remove numbers
    kalimat = [re.sub('\w*\d\w*', '', x) for x in kalimat]
    # Remove non ascii
    kalimat = [re.sub('[^\x00-\x7f]', '', x) for x in kalimat]
    # Remove new line
    kalimat = [re.sub('\n', '', x) for x in kalimat]
    # Remove extra spaces
    kalimat = [re.sub(' +', ' ', x) for x in kalimat]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    kalimat = [stemmer.stem(x) for x in kalimat]

    stopword = stopwords.words('indonesian')
    kalimat = [' '.join([word for word in x.split() if word not in (stopword)]) for x in kalimat]

    # TF-IDF
    kalimat_tfidf = tfidf.transform(kalimat)

    # Predict
    y_pred = xgb_tfidf.predict(kalimat_tfidf)

    # Decode label
    y_pred = le.inverse_transform(y_pred)

    # Print the 'response' column that match with the predicted label
    df[df['topic'] == y_pred[0]]['response'].values[0]
    return df[df['topic'] == y_pred[0]]['response'].values[0]

# Create input text
input_text = st.text_input('Masukkan Pertanyaan Anda')

# Create button
btn = st.button('Submit')

# Create output text
if btn:
    # predict_text = 
    st.markdown(predict([input_text]), unsafe_allow_html=True)
