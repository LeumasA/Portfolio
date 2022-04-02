

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


#def solution_model():
url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

df = pd.read_json("sarcasm.json")
train_df = df.sample(frac=0.8, random_state=25)
test_df = df.drop(train_df.index)
print(df.head())
# Fitting to the input data
tokenizer = Tokenizer(num_words=700, oov_token='OOV')
tokenizer.fit_on_texts(train_df.head()['headline'])



# Toke00ns
word_index = tokenizer.word_index
print(word_index)

# Generate text sequences
sequences = tokenizer.texts_to_sequences(train_df.head()['headline'])
print(sequences)

padded = pad_sequences(sequences, padding='post')
print(padded)


# Splitting the data into 2/3 as train and 1/3 as test
X_train, X_test, y_train, y_test = train_test_split(train_df['headline'], train_df['is_sarcastic'], test_size=0.33, random_state=42)

# Tokenization
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
testing_sequences = tokenizer.texts_to_sequences(X_test)

# Padding
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)  #
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)  # , maxlen=max_length

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(padded, y_train, epochs=10, validation_data=(testing_padded, y_test))

