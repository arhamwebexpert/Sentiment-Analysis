import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_reviews_from_directory(directory_path, label):
    reviews = []
    labels = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                review = file.read()
                reviews.append(review)
                labels.append(label)
    return reviews, labels

def load_data(base_path, max_words=8000, max_len=500):
    pos_train_reviews, pos_train_labels = load_reviews_from_directory(os.path.join(base_path, 'train/pos'), 1)
    neg_train_reviews, neg_train_labels = load_reviews_from_directory(os.path.join(base_path, 'train/neg'), 0)
    pos_test_reviews, pos_test_labels = load_reviews_from_directory(os.path.join(base_path, 'test/pos'), 1)
    neg_test_reviews, neg_test_labels = load_reviews_from_directory(os.path.join(base_path, 'test/neg'), 0)

    reviews = pos_train_reviews + neg_train_reviews + pos_test_reviews + neg_test_reviews
    labels = pos_train_labels + neg_train_labels + pos_test_labels + neg_test_labels

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, word_index

def load_glove_embeddings(glove_file_path, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_embedding_matrix(word_index, embeddings_index, max_words=8000, embedding_dim=100):
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

base_path = 'aclImdb'  # Base path to your dataset
glove_file_path = 'aclImdb/glove.6B.100d.txt'  # Path to your GloVe embeddings file
embedding_dim = 100

X_train, X_test, y_train, y_test, word_index = load_data(base_path)
embeddings_index = load_glove_embeddings(glove_file_path, embedding_dim)
embedding_matrix = get_embedding_matrix(word_index, embeddings_index)
