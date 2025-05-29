import sys
import io
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from utils.data_preprocessing import load_data, get_embedding_matrix, load_glove_embeddings
from models.cnn_model import create_cnn_model
from models.lstm_model import create_lstm_model
from models.lstm_cnn_model import create_lstm_cnn_model
from utils.train import train_model
from utils.evaluate import evaluate_model

# Ensure stdout uses UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_PATH = 'aclImdb'
GLOVE_FILE_PATH = 'aclImdb/glove.6B.100d.txt'
MAX_WORDS = 8000
MAX_LEN = 500
EMBEDDING_DIM = 100

# Load data and embeddings
X_train, X_test, y_train, y_test, word_index = load_data(BASE_PATH, MAX_WORDS, MAX_LEN)
embeddings_index = load_glove_embeddings(GLOVE_FILE_PATH, EMBEDDING_DIM)
embedding_matrix = get_embedding_matrix(word_index, embeddings_index, MAX_WORDS, EMBEDDING_DIM)

def train_and_evaluate_model(model_type, epochs):
    if model_type == 'cnn':
        model = create_cnn_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, embedding_matrix)
    elif model_type == 'lstm':
        model = create_lstm_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, embedding_matrix)
    elif model_type == 'lstm_cnn':
        model = create_lstm_cnn_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, embedding_matrix)
    
    try:
        history = train_model(model, X_train, y_train, X_test, y_test, epochs=epochs)
        if history is not None:
            accuracy, cm, report = evaluate_model(model, X_test, y_test)
            result_label.config(text=f'Accuracy: {accuracy}\n\nConfusion Matrix:\n{cm}\n\nClassification Report:\n{report}')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def on_train_button_click():
    model_type = model_type_var.get()
    epochs = epochs_entry.get()
    if not model_type:
        messagebox.showerror("Error", "Please select a model type.")
        return
    
    if not epochs.isdigit():
        messagebox.showerror("Error", "Please enter a valid number of epochs.")
        return
    
    train_and_evaluate_model(model_type, int(epochs))

def on_hover(event, widget):
    widget.config(bg="#6CA6CD")

def on_leave(event, widget):
    widget.config(bg="#3C3F41")

# GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis Model Trainer")
root.geometry("800x600")
root.configure(bg="#282C34")


try:
    bg_image = Image.open("back.jpeg")
    bg_image = bg_image.resize((800, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
except Exception as e:
    print(f"Error loading background image: {e}")

# Main Heading
main_heading = tk.Label(root, text="Sentiment Analysis Model Trainer", bg="#3C3F41", fg="#F1F1F1", font=("Helvetica", 24, "bold"))
main_heading.pack(pady=20)

# Model Selection Frame
model_frame = tk.Frame(root, bg="#3C3F41", bd=10)
model_frame.pack(pady=20, padx=20, fill="x")

model_label = tk.Label(model_frame, text="Select Model Type:", bg="#3C3F41", fg="#F1F1F1", font=("Helvetica", 16, "bold"))
model_label.grid(row=0, column=0, sticky="w")

model_type_var = tk.StringVar(value="cnn")
cnn_radio = tk.Radiobutton(model_frame, text="CNN", variable=model_type_var, value="cnn", bg="#3C3F41", fg="#F1F1F1", font=("Helvetica", 14))
cnn_radio.grid(row=1, column=0, sticky="w")
cnn_radio.bind("<Enter>", lambda event: on_hover(event, cnn_radio))
cnn_radio.bind("<Leave>", lambda event: on_leave(event, cnn_radio))

lstm_radio = tk.Radiobutton(model_frame, text="LSTM", variable=model_type_var, value="lstm", bg="#3C3F41", fg="#F1F1F1", font=("Helvetica", 14))
lstm_radio.grid(row=2, column=0, sticky="w")
lstm_radio.bind("<Enter>", lambda event: on_hover(event, lstm_radio))
lstm_radio.bind("<Leave>", lambda event: on_leave(event, lstm_radio))

lstm_cnn_radio = tk.Radiobutton(model_frame, text="LSTM+CNN", variable=model_type_var, value="lstm_cnn", bg="#3C3F41", fg="#F1F1F1", font=("Helvetica", 14))
lstm_cnn_radio.grid(row=3, column=0, sticky="w")
lstm_cnn_radio.bind("<Enter>", lambda event: on_hover(event, lstm_cnn_radio))
lstm_cnn_radio.bind("<Leave>", lambda event: on_leave(event, lstm_cnn_radio))

# Epochs Entry
epochs_label = tk.Label(model_frame, text="Enter Number of Epochs:", bg="#3C3F41", fg="#F1F1F1", font=("Helvetica", 16, "bold"))
epochs_label.grid(row=4, column=0, sticky="w")

epochs_entry = tk.Entry(model_frame, bg="#ffffff", font=("Helvetica", 14))
epochs_entry.grid(row=5, column=0, sticky="w")

# Train Button
train_button = tk.Button(root, text="Train and Evaluate", command=on_train_button_click, bg="#4CAF50", fg="white", font=("Helvetica", 16, "bold"), padx=10, pady=5)
train_button.pack(pady=20)

# Results Display
result_label = tk.Label(root, text="", justify=tk.LEFT, bg="#282C34", fg="#F1F1F1", font=("Helvetica", 14), bd=10, relief="sunken")
result_label.pack(pady=20, padx=20, fill="both", expand=True)

root.mainloop()
