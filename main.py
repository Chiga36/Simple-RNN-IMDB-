import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

words_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in words_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

## Step 2: Helper Functions
# Function to decode reviews
def decode_review(encode_reviews):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encode_reviews])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encode_review = [words_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encode_review],maxlen=500)
    return padded_review

## Designing STreamlit app:
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie to classify it as positive or negative')

# User Input
user_input = st.text_area('Movie review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the Result
    st.write(f'Sentiment: {sentiment}')
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write('Please ENter the movie review')