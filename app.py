import nltk
import os
import datetime
import csv
import ssl
import streamlit as st
import numpy as np
import json
import random
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

model = load_model('chatbot_model.h5')
with open('expanded_medical.json', 'r') as file:
    intents = json.load(file)
    
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
words = [...] 
classes = [...] 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intent, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
st.title("Medical Chatbot")
st.write("Ask me anything about medical issues!")

user_input = st.text_input("Your Question:")

if st.button("Get Response"):
    if user_input:
        intents_result = predict_class(user_input)
        if intents_result:
            response = get_response(intents_result[0]['intent'], intents)
            st.write(f"**Chatbot:** {response}")
        else:
            st.write("**Chatbot:** Sorry, I didn't understand that.")
    else:
        st.write("Please enter a question.")
