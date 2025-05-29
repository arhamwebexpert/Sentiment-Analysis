from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_lstm_model(input_length, max_words, embedding_dim, embedding_matrix):
    model = Sequential()
    embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=input_length, trainable=False)
    embedding_layer.build((None,))  # Build the layer to be able to set weights
    embedding_layer.set_weights([embedding_matrix])
    
    model.add(embedding_layer)
    model.add(LSTM(200))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
