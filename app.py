# -*- coding: utf-8 -*-
import pandas as pd
from flask import Flask, jsonify, request
import streamlit as st
import pandas as pd
import numpy as np
import re, string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import sys
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

model = pickle.load(open('model1.pkl','rb'))

def preprocess(sentence):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    def cleanHtml(sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext
    
    def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned
    
    def keepAlpha(sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent
    
    
    def removeStopWords(re_stop_words, sentence):
#        global re_stop_words
        return re_stop_words.sub(" ", sentence)
    
    
    def Stemmer(sentence):
        stem_list = [sn_stemmer.stem(word) for word in sentence.split()]
        return ' '.join(stem_list)

    sentence = cleanHtml(sentence)
    sentence = cleanPunc(sentence)
    sentence = keepAlpha(sentence)
    
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    sentence = removeStopWords(re_stop_words, sentence)
    
    sn_stemmer = SnowballStemmer('english')
    sentence = Stemmer(sentence)
    return sentence

def predict(comment):

    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']
    label_array = np.array(categories).reshape(1,6)
    output_array = np.zeros((1,len(categories)))
    pre_Sentence = preprocess(comment)
    bow_input_comment = model['vec'].transform([pre_Sentence])
    for i, category  in enumerate(categories):
        r = model['r'][category]
        modified_input = bow_input_comment.multiply(r)
        output_array[:, i] = model['m'][category].predict_proba(modified_input)[:,1]
#     output_array>0.9
    return label_array[output_array>0.5]

  


def main():
    st.title('Toxic Comment Classifier')
    comment_input = st.text_area("Enter the Comment to Classify")
    
    result = ''   
    if st.button('Predict'):
        result = predict(comment_input)
    
    if len(result) > 0:
        show_result = "Given Comment :" + comment_input + " contains " + ', '.join(result)
        st.success(show_result)
    else:    
        st.success("Given Comment :" + comment_input + " contains " + "No toxic elements")

if __name__=='__main__':
    main()