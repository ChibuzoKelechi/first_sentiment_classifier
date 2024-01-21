# Import dependencies
import nltk 
import streamlit as st 
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from selenium import webdriver
from bs4 import BeautifulSoup
import time 

# Tokenize sentence/input
def token_sentence(sent):
    return ({word: True for word in nltk.word_tokenize(sent.lower())})

# List variables to store text data
pos_tweets = []
neg_tweets = []

# Extract data from files
with open("sentiment_nlp_app/pos_tweets.txt") as f:
    for i in f: 
        pos_tweets.append([token_sentence(i), 'positive'])

with open('sentiment_nlp_app/neg_tweets.txt') as nf:
    for n in nf:
        neg_tweets.append([token_sentence(n), 'negative'])
        

# Split data
train_data = (
    pos_tweets[:int((.9)*len(pos_tweets))] + neg_tweets[:int((.9) * len(neg_tweets))] 
)

test_data = (
    pos_tweets[:int((.1)*len(pos_tweets))] + neg_tweets[:int((.1)*len(neg_tweets))] 
)

# Initialize model and classifier function
classifier = NaiveBayesClassifier.train(train_data)
class_accuracy = accuracy(classifier, test_data)


def classify_sentence(input):
    m_accuracy = f'{int(class_accuracy * 100)}%'
    sentiment_class = classifier.classify(token_sentence(input))
    return f"Connotation: {sentiment_class}. {m_accuracy} sure"


# Streamlit app layout/components
st.title('Basic Sentiment classification')
st.write('''
         This is a simple ML web app to detect positive/negative sentiment in the given text.
         It is an application of Natural language processing with NLTK, and web scraping(Tweets). 
    ''')

st.image('sentiment_nlp_app/nlp.jpg')

st.subheader('Direct input')
input = st.text_input('Type in sentence')
result = classify_sentence(input) 

if input:
    st.write(result)
    
    
# Tweet scraping functionality
st.subheader('Tweet URL input')
tweet_url = st.text_input('Paste tweet URL to extract tweet')

def scrape_tweet_url(url):
        
    driver = webdriver.Firefox() # Initialize web driver
    driver.minimize_window()
    driver.get(url)
    time.sleep(5)
    
    resp = driver.page_source
    driver.close()
    
    tweet_soup = BeautifulSoup(resp, 'html.parser')
    
    try:
        # Get Tweet text with bs4 
        tweet_source = tweet_soup.find("div",{"data-testid":"tweetText"})
        tweet_text = tweet_source.find('span', class_='css-1qaijid r-bcqeeo r-qvutc0 r-poiln3').text

    except:
        tweet_text = None
        st.write('404. Tweet Not Found')
        
    return tweet_text

def scrape_and_classify(scrape_function):
    try: # Exception handling
        if tweet_url:
          tweet = scrape_function(tweet_url)
          tweet_sentiment = classify_sentence(tweet)
          st.write(f'Tweet text: {tweet}')
          st.write(tweet_sentiment)
    
    except:
        st.write('404 Not found. Error occured in retrieving tweet')
        # Fallback message(In case of error)
    
scrape_and_classify(scrape_tweet_url) # Final function


# Footer
st.markdown("<hr style='border: 1px dashed #ddd; margin: 2rem;'>", unsafe_allow_html=True) #Horizontal line

st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        Project by <a href="https://github.com/ChibuzoKelechi" target="_blank" style="color: white; font-weight: bold; text-decoration: none;">
         kelechi_tensor</a>
    </div>
    
    <div style="text-align: center; padding: 1rem;">
        Data from <a href="https://kaggle.com" target="_blank" style="color: lightblue; font-weight: bold; text-decoration: none;">
         Kaggle</a>
    </div>
""",
unsafe_allow_html=True)

# Peace Out :)