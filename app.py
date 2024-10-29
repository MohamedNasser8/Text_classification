import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

stemmer = SnowballStemmer("english", ignore_stopwords=True)

stop_words = set(stopwords.words("english"))

remove_stopwords = True

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    # Stem the tokens
    # tokens = [stemmer.stem(word) for word in tokens]
    
    tokens = list(dict.fromkeys(tokens))
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    
    return preprocessed_text


# Load the model
model = joblib.load('model.pkl')

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title of the app
st.title("Simple Text Classification App")

# Input fields
input_data = st.text_input("Enter input data")

if st.button("Predict"):
    input_data_list = [input_data]
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data_list, columns=['text'])
    
    X_tfidf = tfidf_vectorizer.transform(input_df['text'])
    
    # Predict using the model
    prediction = model.predict(X_tfidf)
    
    # Display the prediction
    st.write(f"Prediction: {prediction}")
