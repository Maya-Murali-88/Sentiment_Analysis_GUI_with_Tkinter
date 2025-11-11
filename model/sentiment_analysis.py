import re
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Setup NLTK resources
def setup_nltk():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    pos_taggers = ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger']
    for res in resources:
        nltk.download(res, quiet=True)
    for tagger in pos_taggers:
        try:
            nltk.download(tagger, quiet=True)
            break
        except:
            continue

setup_nltk()

stop_words = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS for lemmatization"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text: str) -> str:
    """Clean, remove stopwords, and lemmatize text"""
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
        lmtzr.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in stop_words
    ]
    return " ".join(text)
