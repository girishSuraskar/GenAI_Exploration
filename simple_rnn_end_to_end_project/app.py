import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb




# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_end_to_end_project\simple_rnn_model_imdb.h5')
# Prepare word index
# Load the IMDB dataset
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {v: k for k, v in word_index.items()}

# Decode function
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Preprocess function
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)

def predict_sentiment(review):
    # Preprocess the review text
    encoded_review = preprocess_text(review)
    
    # Make prediction
    prediction = model.predict(encoded_review)
    
    # Determine sentiment and score
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
