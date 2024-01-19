import nltk 
import streamlit as st 
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy 
from bs4 import BeautifulSoup
import requests 


def token_sentence(sent):
    return ({word: True for word in nltk.word_tokenize(sent)})

tokentxt = token_sentence('These are the angels.')

# print(tokentxt)

pos_tweets = []
neg_tweets = []

with open("st_sandbox/pos_tweets.txt") as f:
    for i in f: 
        pos_tweets.append([token_sentence(i), 'positive'])

with open('st_sandbox/neg_tweets.txt') as nf:
    for n in nf:
        neg_tweets.append([token_sentence(n), 'negative'])

len(pos_tweets)

train_data = pos_tweets[:int((.9)*len(pos_tweets))] + neg_tweets[:int((.9)*len(neg_tweets))]

test_data = pos_tweets[:int((.1)*len(pos_tweets))] + neg_tweets[:int((.1)*len(neg_tweets))]

classifier = NaiveBayesClassifier.train(train_data)

classifier.show_most_informative_features()
class_accuracy = accuracy(classifier, test_data)

first_text = 'Poor cavs....all that hard work.'
pos_text = 'Now I can breathe haha '

def classify_sentence(input):
    m_accuracy = f'{int(class_accuracy * 100)}%'
    sentiment_class = classifier.classify(token_sentence(input))
    return f"Connotation: {sentiment_class} with accuracy of {m_accuracy}"


st.title('Basic Sentiment classification')
st.write('''
         This is a simple ML web app to detect positive/negative sentiment in the given text.
         It is an application of Natural language processing with NLTK, and web scraping(Tweets). 
    ''')

input = st.text_input('text_in')
result = classify_sentence(input) 

st.write(result)