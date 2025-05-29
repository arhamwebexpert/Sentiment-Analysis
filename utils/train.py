from keras.callbacks import ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
    checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True, mode='min')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

