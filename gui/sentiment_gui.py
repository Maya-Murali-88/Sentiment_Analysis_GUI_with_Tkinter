import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# Load the model and vectorizer
with open('../model/sentiment_analysis_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../model/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


# Load dataset
reviews = pd.read_csv('../data/customer_reviews.csv', encoding='ISO8859-1')
reviews.rename(columns={'detailed_review': 'Detailed Review'}, inplace=True)


# Preprocessing function (same as model)
stop_words = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()

def preprocess_review(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().split()
    tagged = pos_tag(text)
    text = [
        lmtzr.lemmatize(word, wordnet.NOUN if tag is None else wordnet.NOUN)
        for word, tag in tagged
        if word not in stop_words
    ]
    return " ".join(text)


# Initialize Tkinter
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x400")
root.configure(bg="white") 

# Center everything using a Frame
main_frame = tk.Frame(root, bg="white")
main_frame.pack(expand=True)


# Load emojis
happy_emoji = ImageTk.PhotoImage(Image.open('../assets/happy.png').resize((150, 150)))
neutral_emoji = ImageTk.PhotoImage(Image.open('../assets/neutral.png').resize((150, 150)))
angry_emoji = ImageTk.PhotoImage(Image.open('../assets/angry.png').resize((150, 150)))


# Labels for review and emoji
emoji_label = tk.Label(main_frame, bg="white", bd=0)
emoji_label.pack(pady=20)

review_label = tk.Label(main_frame, wraplength=400, justify="center", bg="white", font=("Arial", 12))
review_label.pack(pady=10)


# Keep track of current index
current_index = [0]


# Function to update review & emoji
def update_feedback(index):
    review_text = reviews['Detailed Review'].iloc[index]
    processed_review = preprocess_review(review_text)
    X_review = vectorizer.transform([processed_review])
    predicted_sentiment = model.predict(X_review)[0]  # -1,0,1

    emoji_map = {-1: angry_emoji, 0: neutral_emoji, 1: happy_emoji}
    emoji_image = emoji_map.get(predicted_sentiment, neutral_emoji)

    # Update GUI
    review_label.config(text=review_text)
    emoji_label.config(image=emoji_image)
    emoji_label.image = emoji_image


# Button to go to next review
def next_review():
    current_index[0] = (current_index[0] + 1) % len(reviews)
    update_feedback(current_index[0])

next_button = ttk.Button(main_frame, text="Next Review", command=next_review)
next_button.pack(pady=20)


# Initialize first review
update_feedback(current_index[0])

root.mainloop()
