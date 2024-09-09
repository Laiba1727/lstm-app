import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
import nltk
import streamlit as st 

st.title("Sentiment Analysis of IMDB Reviews")
st.write("Upload your IMDB dataset (CSV) to analyze sentiment.")
uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

 
def remove_tags(string):
        result = re.sub(r'<.*?>', '', string)
        result = re.sub(r'https?://\S+', '', result)
        result = re.sub(r'[^a-zA-Z0-9\s]', '', result)
        result = result.lower()
        return result


st.write("Removing tags and unwanted characters...")
data['review'] = data['review'].apply(lambda cw: remove_tags(cw))
    
    
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

st.write("Removing stopwords...")
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

   
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st

st.write("Lemmatizing text...")
data['review'] = data['review'].apply(lemmatize_text)
    
 
st.write("Sample cleaned data:")
st.write(data.head())

pos = data[data['sentiment'] == 'positive'].shape[0]
neg = data[data['sentiment'] == 'negative'].shape[0]

st.write(f"Percentage of reviews with positive sentiment: {pos / data.shape[0] * 100:.2f}%")
st.write(f"Percentage of reviews with negative sentiment: {neg / data.shape[0] * 100:.2f}%")

  
st.write("Encoding labels...")
reviews = data['review'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

    
vocab_size = 3000
embedding_dim = 100
max_length = 200
padding_type = 'post'
trunc_type = 'post'
oov_tok = ''

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)
padded_reviews = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    
train_sentences, test_sentences, train_labels, test_labels = train_test_split(padded_reviews, encoded_labels, test_size=0.2, random_state=42)

    
st.write("Building the LSTM model...")
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

st.write(model.summary())

    
num_epochs = 5
history = model.fit(train_sentences, train_labels, epochs=num_epochs, validation_split=0.1, verbose=1)

 
st.write("Making predictions...")
prediction = model.predict(test_sentences)

pred_labels = (prediction >= 0.5).astype(int)

accuracy = accuracy_score(test_labels, pred_labels)
st.write(f"Accuracy on the test set: {accuracy:.2f}")

st.write("Classification report:")
st.text(classification_report(test_labels, pred_labels))

