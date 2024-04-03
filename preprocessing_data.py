
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import string
from string import digits
from pprint import pprint
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


"""# Data Preprocessing

### Understanding Problem Statement

---

Our dataset comprises 1700 unique articles and it's title sourced from various outlets. Each article is accompanied by a label indicating its topic, such as sports, science, politics, and more. This label serves as the target variable for classification based on the content of the articles.
"""
print("Data is preprocessing")
df = pd.read_csv("data.csv")

"""We can observe lots of noise at first mail like extra spaces, many hyphen marks, different cases, and many more.

### Missing Value
"""

df.drop_duplicates(inplace=True)
# Remove "Enlarge this image" from the content column
df['content'] = df['content'].str.replace('Enlarge this image', '')
df['content'] = df['content'].str.replace('hide caption', '')
df['content'] = df['content'].str.replace('toggle caption', '')
df['content'] = df['content'].str.replace('caption toggle', '')
df = df.dropna()
df = pd.DataFrame(df)
df.drop("link", axis=1, inplace=True)
"""### Remove URL"""
def remove_url(data):
    url_removed = []
    for line in data:
        url_removed.append(re.sub('http[s]?://\S+', '', line))
    return url_removed
df = df.apply(remove_url)

"""### Remove hashtag"""

def remove_hashtag(data):
    hashtag_removed = []
    translator = str.maketrans('#', ' '*len('#'), '')
    for line in data:
        hashtag_removed.append(line.translate(translator))
    return hashtag_removed

df = df.apply(remove_hashtag)

"""### Expand Contractions

Contraction is the shortened form of a word like don’t stands for do not, aren’t stands for are not
"""

contractions_dict = {"ain't": "are not","'s":" is","aren't": "are not", "n't" : "not"}
# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)
# Expanding Contractions in the reviews
df['content']=df['content'].apply(lambda x:expand_contractions(x))
df['title']=df['title'].apply(lambda x:expand_contractions(x))

"""Lower case"""

df['title'] = df['title'].str.lower()
df['content'] = df['content'].str.lower()

"""### Remove punctuation"""
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["title"]= df["title"].apply(lambda text: remove_punctuation(text))
df["content"]= df["content"].apply(lambda text: remove_punctuation(text))
df["title"] = df["title"].apply(lambda x: re.sub(r"[–”“—’‘]", "", x))
df["content"] = df["content"].apply(lambda x: re.sub(r"[–”“—’‘]", "", x))

"""### Tokenization"""

def tokenize_sentence(data):
  tokenized_docs = []
  for line in data:
    tokenized_docs.append(word_tokenize(line))
  return tokenized_docs

df = df.apply(tokenize_sentence)


"""### Remove words and digits containing digits"""
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

"""### Remove stopwords"""
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

dt = df.copy()

"""### Stemming and Lemmatization

Stemming is a process to reduce the word to its root stem for example run, running, runs, runed derived from the same word as run. basically stemming do is remove the prefix or suffix from word like ing, s, es, etc...

Lemmatization is similar to stemming, used to stem the words into root word but differs in working. Actually, Lemmatization is a systematic way to reduce the words into their lemma by matching them with a language dictionary.
"""


def apply_stemmer(data):
  stemmed_docs = []
  stemmer = PorterStemmer()
  for doc in data:
    stemmed_docs.append([stemmer.stem(plural) for plural in doc])
  return stemmed_docs
df = df.apply(apply_stemmer)

def lemmatize_words(data):
    lemmatized_docs = []
    lemmatizer = WordNetLemmatizer()
    for doc in data:
        lemmatized_docs.append([lemmatizer.lemmatize(word) for word in doc])
    return lemmatized_docs
df = df.apply(lemmatize_words)
"""# Feature Extraction
### Bag of words
Bag of words approach involves breaking down a piece of text into individual words, and then representing the text as a frequency distribution of those words. In other words, we’re creating a “bag” of all the words in the text, without any regard for their order or context, and then counting how many times each word appears in that bag. This simple yet effective method allows us to extract meaningful insights from large volumes of text data, such as identifying the most frequent words, analyzing sentiment, or even predicting future trends.
"""

combined_paragraphs = []
# create a full paragraph of new
for index, row in df.iterrows():
    title_text = ' '.join(row['title'])
    content_text = ' '.join(row['content'])
    combined_text = title_text + ' ' + content_text
    combined_paragraphs.append(combined_text)
df['paragraph'] = combined_paragraphs
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(combined_paragraphs)
bag_of_words = X.toarray()
df['bag_of_words'] = list(bag_of_words)
df['category'] = df['category'].apply(lambda x: x[0])

# Encoding the category column
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

"""##### TF-IDF

Word counts are very basic.

An alternative is to calculate word frequencies.

* Term Frequency: This summarizes how often a given word appears within a document.

* Inverse Document Frequency: This downscales words that appear a lot across documents.

TF-IDF are word frequency scores that try to highlight words that are more frequent in a document but not across documents.If we already have a learned CountVectorizer, we can use it with a TfidfTransformer to just calculate the inverse document frequencies and start encoding documents. The same create, fit, and transform process is used as with the CountVectorizer.
"""

best_tfidf_max_df = 0.7735395966421682 #this is the best parameter after tuning
best_tfidf_ngram_range = (1, 4)
best_tfidf_max_features = 22251
tfidf_vectorizer = TfidfVectorizer(max_df=best_tfidf_max_df,
                                   ngram_range=best_tfidf_ngram_range,
                                   max_features=best_tfidf_max_features)
X = tfidf_vectorizer.fit_transform(combined_paragraphs)
tf = X.toarray()
df['TF-IDF'] = list(tf)


"""### Prediction Base Embedding : Word2Vec using gensim model"""



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
df['word2vec'] = df['paragraph'].apply(lambda x: get_paragraph_embedding(x, word2vec_model))
df['combined_text'] = df.apply(lambda row: row['title'] + row['content'], axis=1)

