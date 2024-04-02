import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from string import digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import gensim
import gensim.corpora as corpora
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from gensim.utils import simple_preprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
import optuna
from keras.initializers import Constant
from keras.models import Model
from keras.layers import *

df = pd.read_csv("data.csv")
df.drop_duplicates(inplace=True)
# Remove "Enlarge this image" from the content column
df['content'] = df['content'].str.replace('Enlarge this image', '')
df['content'] = df['content'].str.replace('hide caption', '')
df['content'] = df['content'].str.replace('toggle caption', '')
df['content'] = df['content'].str.replace('caption toggle', '')
# Display the DataFrame with the modified content column

#
# Preprocessing
#

# drop n/a
df = df.dropna()
df = pd.DataFrame(df)

# drop urls 
def remove_url(data):
    url_removed = []
    for line in data:
        url_removed.append(re.sub('http[s]?://\S+', '', line))
    return url_removed

df = df.apply(remove_url)

# drop hashtags
def remove_hashtag(data):
    hashtag_removed = []
    translator = str.maketrans('#', ' '*len('#'), '')
    for line in data:
        hashtag_removed.append(line.translate(translator))
    return hashtag_removed

df = df.apply(remove_hashtag)

# remove punctuations
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["title"]= df["title"].apply(lambda text: remove_punctuation(text))
df["content"]= df["content"].apply(lambda text: remove_punctuation(text))
df["title"] = df["title"].apply(lambda x: re.sub(r"[–”“—’‘]", "", x))
df["content"] = df["content"].apply(lambda x: re.sub(r"[–”“—’‘]", "", x))
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
concec = '“'
remove_punctuation(concec)
concec

# remove contractions
contractions_dict = {"ain't": "are not","'s":" is","aren't": "are not", "n't" : "not"}
# Regular expression for finding contractions
# "'d" :"had or would"
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)
# Expanding Contractions in the reviews
df['content']=df['content'].apply(lambda x:expand_contractions(x))
df['title']=df['title'].apply(lambda x:expand_contractions(x))

# apply lowercase 
df['title'] = df['title'].str.lower()
df['content'] = df['content'].str.lower()

# apply tokenizations
def tokenize_sentence(data):
  tokenized_docs = []

  for line in data:
    tokenized_docs.append(word_tokenize(line))

  return tokenized_docs

df = df.apply(tokenize_sentence)

# removing digits
from nltk.stem import WordNetLemmatizer
def remove_digits(data):
    digit_removed = []

    for doc in data:
        temp = []
        for word in doc:
            # Check if the word contains only digits
            if not word.isdigit():
                temp.append(word)
        digit_removed.append(temp)

    return digit_removed

df = df.apply(remove_digits)

# removing stopwords
def remove_stopwords(data):

  stopword_removed = []

  stop_words = set(stopwords.words('english'))
  more_stop = ["say", "said", "says", "u"]
  for word in more_stop:
    stop_words.add(word)

  for doc in data:
    temp = []
    for word in doc:
      if word not in stop_words:
        temp.append(word)

    stopword_removed.append(temp)

  return stopword_removed

df = df.apply(remove_stopwords)

# apply stemming
def apply_stemmer(data):
  stemmed_docs = []

  stemmer = PorterStemmer()

  for doc in data:
    stemmed_docs.append([stemmer.stem(plural) for plural in doc])

  return stemmed_docs

df = df.apply(apply_stemmer)

# apply lemmatizing
def lemmatize_words(data):
    lemmatized_docs = []
    lemmatizer = WordNetLemmatizer()
    for doc in data:
        lemmatized_docs.append([lemmatizer.lemmatize(word) for word in doc])
    return lemmatized_docs
df = df.apply(lemmatize_words)


df = df.drop("link", axis=1)
df['combined_text'] = df.apply(lambda row: row['title'] + row['content'], axis=1)
data_words = df['combined_text']
df['category'] = df['category'].apply(lambda x: x[0])
label_encoder = LabelEncoder()

df['label'] = label_encoder.fit_transform(df['category'])

combined_paragraphs = []
# create a full paragraph of new
for index, row in df.iterrows():
    title_text = ' '.join(row['title'])
    content_text = ' '.join(row['content'])
    combined_text = title_text + ' ' + content_text
    combined_paragraphs.append(combined_text)

df['paragraph'] = combined_paragraphs

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Define data
paragraphs = df['paragraph']
tokenized_paragraphs = [simple_preprocess(paragraph) for paragraph in paragraphs]

# Define parameters for Word2Vec model
vector_size = 300
window = 5
min_count = 5


# Train Word2Vec model with specified parameters
word2vec_model = Word2Vec(sentences=tokenized_paragraphs, vector_size=vector_size, window=window, min_count=min_count, sg=0)

# Function to obtain paragraph embeddings using the trained Word2Vec model
def get_paragraph_embedding(paragraph, model):
    # Tokenize and preprocess the paragraph
    tokens = simple_preprocess(paragraph)

    # Filter out tokens that are not in the vocabulary of the Word2Vec model
    tokens_in_vocab = [token for token in tokens if token in model.wv]

    # If no tokens are in the vocabulary, return None
    if not tokens_in_vocab:
        return None

    # Calculate the average word embedding for the tokens in the paragraph
    paragraph_embedding = np.mean([model.wv[token] for token in tokens_in_vocab], axis=0)
    return paragraph_embedding

# Apply the function to each paragraph in the DataFrame
df['skip_gram'] = df['paragraph'].apply(lambda x: get_paragraph_embedding(x, word2vec_model))

x_train, x_test, y_train, y_test = train_test_split(df['skip_gram'], df['label'], test_size=500, shuffle=False)
x_train = np.array([np.array(x) for x in x_train])
x_test = np.array([np.array(x) for x in x_test])


# Instantiate SVM model with specified hyperparameters
svm_model = SVC(C=0.4, kernel='linear', gamma='scale')

# Fit the model to the training data
svm_model.fit(x_train, y_train)

# Make predictions on test data
y_pred = svm_model.predict(x_test)

# Calculate accuracy on test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Data:", accuracy)

# building Gradio demo
import gradio as gr
def make_prediction(text):
    tokens = simple_preprocess(text)
    tokens_in_vocab = [token for token in tokens if token in word2vec_model.wv]
    embedding = get_paragraph_embedding(text, word2vec_model)
    new_text_embedding = np.mean([word2vec_model.wv[token] for token in tokens_in_vocab], axis=0)
    new_text_embedding = [new_text_embedding]
    predictions = svm_model.predict(new_text_embedding)  # Make predictions
    if predictions == 4:
        return "Sport"
    elif predictions == 3:
        return "Science"
    elif predictions == 2:
        return "Politics"
    elif predictions == 1:
        return "Health"
    else:
        return "Business"
    return predictions

news = gr.Textbox()
output = gr.Textbox()
app = gr.Interface(
    fn=make_prediction,
    inputs="text",
    outputs="text",
)
print(app.launch(share=True))