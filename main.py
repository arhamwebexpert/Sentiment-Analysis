import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.data_preprocessing import load_data, get_embedding_matrix, load_glove_embeddings
from models.cnn_model import create_cnn_model
from models.lstm_model import create_lstm_model
from models.lstm_cnn_model import create_lstm_cnn_model
from utils.train import train_model
from utils.evaluate import evaluate_model

# Page configuration
st.set_page_config(
    page_title="🎭 Sentiment Analysis Suite",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Constants
BASE_PATH = 'aclImdb'
GLOVE_FILE_PATH = 'aclImdb/glove.6B.100d.txt'
MAX_WORDS = 8000
MAX_LEN = 500
EMBEDDING_DIM = 100

@st.cache_data
def load_initial_data():
    """Load data and embeddings on startup"""
    try:
        X_train, X_test, y_train, y_test, word_index = load_data(BASE_PATH, MAX_WORDS, MAX_LEN)
        embeddings_index = load_glove_embeddings(GLOVE_FILE_PATH, EMBEDDING_DIM)
        embedding_matrix = get_embedding_matrix(word_index, embeddings_index, MAX_WORDS, EMBEDDING_DIM)
        # Auto-load existing model if it exists
        import os
        current_model = None
        model_type = None
        if os.path.exists('model.keras'):
            from tensorflow.keras.models import load_model
            current_model = load_model('model.keras')
            model_type = "loaded"  # or detect type from model
            print("Loaded existing model.keras")
        return X_train, X_test, y_train, y_test, word_index, embeddings_index, embedding_matrix
    except Exception as e:
        st.error(f"Initialization Error: Failed to load data: {e}")
        return None, None, None, None, None, None, None

def preprocess_text(text, word_index):
    """Preprocess text for prediction"""
    words = text.lower().split()
    sequences = []
    for word in words:
        if word in word_index:
            sequences.append(word_index[word])
    return pad_sequences([sequences], maxlen=MAX_LEN)

def create_sentiment_gauge(confidence, sentiment):
    """Create a beautiful gauge chart for sentiment"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Sentiment: {sentiment}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea" if sentiment == "Positive" else "#f5576c"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccb"},
                {'range': [50, 100], 'color': "#90ee90"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_training_progress_chart(history):
    """Create training progress visualization"""
    if history is None:
        return None
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Accuracy', 'Model Loss'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history.history['accuracy'], 
                  name='Training Accuracy', line=dict(color='#667eea')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history.history['val_accuracy'], 
                  name='Validation Accuracy', line=dict(color='#764ba2')),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history.history['loss'], 
                  name='Training Loss', line=dict(color='#f093fb')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history.history['val_loss'], 
                  name='Validation Loss', line=dict(color='#f5576c')),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    return fig

def load_model_util():
    """Load a pre-trained model"""
    try:
        from tensorflow.keras.models import load_model
        import os
        if os.path.exists('model.keras'):
            model = load_model('model.keras')
            st.info("Model loaded successfully!")
            return model, "loaded"
        else:
            st.error("model.keras file not found!")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Advanced Sentiment Analysis Suite</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading dataset and embeddings..."):
            data = load_initial_data()
            if data[0] is not None:
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.word_index, st.session_state.embeddings_index, st.session_state.embedding_matrix = data
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            else:
                st.error("❌ Failed to load data. Please check your file paths.")
                return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model status
        if st.session_state.model is not None:
            st.success(f"✅ Model Ready: {st.session_state.model_type.upper()}")
        else:
            st.warning("⚠️ No model trained")
        
        st.markdown("---")
        
        # Dataset info
        st.markdown("### 📊 Dataset Info")
        if st.session_state.data_loaded:
            st.metric("Training Samples", len(st.session_state.X_train))
            st.metric("Test Samples", len(st.session_state.X_test))
            st.metric("Vocabulary Size", MAX_WORDS)
            st.metric("Max Sequence Length", MAX_LEN)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.model = None
            st.session_state.model_type = None
            st.session_state.training_history = None
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "🎓 Model Training", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.markdown("## 🔮 Sentiment Prediction")
        
        if st.session_state.model is None:
            st.warning("⚠️ Please train a model first using the 'Model Training' tab.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Enter your text:")
                user_text = st.text_area(
                    "Text to analyze",
                    height=150,
                    placeholder="Enter your text here for sentiment analysis..."
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    predict_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
                with col_btn2:
                    clear_btn = st.button("🗑️ Clear", use_container_width=True)
                
                if clear_btn:
                    st.rerun()
                
                if predict_btn and user_text.strip():
                    with st.spinner("🤔 Analyzing sentiment..."):
                        try:
                            processed_text = preprocess_text(user_text, st.session_state.word_index)
                            prediction = st.session_state.model.predict(processed_text)
                            confidence = float(prediction[0][0])
                            
                            if confidence > 0.5:
                                sentiment = "Positive"
                                confidence_pct = confidence * 100
                                emoji = "😊"
                            else:
                                sentiment = "Negative"
                                confidence_pct = (1 - confidence) * 100
                                emoji = "😞"
                            
                            # Display results
                            st.markdown("### 📊 Analysis Results")
                            
                            col_res1, col_res2, col_res3 = st.columns(3)
                            with col_res1:
                                st.metric("Sentiment", f"{emoji} {sentiment}")
                            with col_res2:
                                st.metric("Confidence", f"{confidence_pct:.1f}%")
                            with col_res3:
                                st.metric("Raw Score", f"{confidence:.4f}")
                            
                            # Gauge chart
                            gauge_fig = create_sentiment_gauge(confidence_pct, sentiment)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            # Detailed results
                            with st.expander("📋 Detailed Analysis"):
                                st.write(f"**Input Text:** {user_text}")
                                st.write(f"**Model Used:** {st.session_state.model_type.upper()}")
                                st.write(f"**Processing:** Text was tokenized and padded to {MAX_LEN} tokens")
                                st.write(f"**Interpretation:** Values > 0.5 indicate positive sentiment")
                                
                        except Exception as e:
                            st.error(f"❌ Error making prediction: {e}")
            
            with col2:
                st.markdown("### 💡 Tips")
                st.info("""
                **For better results:**
                - Use complete sentences
                - Include context
                - Avoid excessive punctuation
                - Try different text lengths
                """)
                
                st.markdown("### 📝 Example Texts")
                examples = [
                    "I love this movie! It's absolutely fantastic.",
                    "This product is terrible, waste of money.",
                    "The weather is okay today, nothing special.",
                    "Amazing service, highly recommended!"
                ]
                
                for i, example in enumerate(examples):
                    if st.button(f"📄 Example {i+1}", key=f"example_{i}"):
                        st.session_state.example_text = example
    
    with tab2:
        st.markdown("## 🎓 Model Training Center")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🏗️ Model Configuration")
            
            model_type = st.selectbox(
                "Select Model Architecture:",
                ["cnn", "lstm", "lstm_cnn"],
                format_func=lambda x: {
                    "cnn": "🔄 CNN - Convolutional Neural Network",
                    "lstm": "🔗 LSTM - Long Short-Term Memory", 
                    "lstm_cnn": "🚀 LSTM+CNN - Hybrid Architecture"
                }[x]
            )
            
            epochs = st.slider("Number of Epochs:", 1, 50, 10)
            
            # Model info
            model_info = {
                "cnn": "Fast training, good for pattern recognition in text",
                "lstm": "Great for sequential data, captures long-term dependencies",
                "lstm_cnn": "Combines both architectures for best performance"
            }
            
            st.info(f"ℹ️ {model_info[model_type]}")
        
        with col2:
            st.markdown("### 📊 Training Status")
            
            if st.session_state.model is not None:
                st.success(f"✅ Current Model: {st.session_state.model_type.upper()}")
            else:
                st.warning("⚠️ No model trained yet")
            
            # Training button
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner(f"🏋️ Training {model_type.upper()} model for {epochs} epochs..."):
                    try:
                        # Create progress bars
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Create model
                        if model_type == 'cnn':
                            model = create_cnn_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, st.session_state.embedding_matrix)
                        elif model_type == 'lstm':
                            model = create_lstm_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, st.session_state.embedding_matrix)
                        elif model_type == 'lstm_cnn':
                            model = create_lstm_cnn_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, st.session_state.embedding_matrix)
                        
                        status_text.text("🏗️ Model created, starting training...")
                        progress_bar.progress(25)
                        
                        # Train model
                        history = train_model(model, st.session_state.X_train, st.session_state.y_train, 
                                            st.session_state.X_test, st.session_state.y_test, epochs=epochs)
                        
                        progress_bar.progress(75)
                        status_text.text("📊 Evaluating model...")
                        
                        if history is not None:
                            # Evaluate model
                            accuracy, cm, report = evaluate_model(model, st.session_state.X_test, st.session_state.y_test)
                            
                            # Save to session state
                            st.session_state.model = model
                            st.session_state.model_type = model_type
                            st.session_state.training_history = history
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Training completed!")
                            
                            st.success(f"🎉 Training completed! Final accuracy: {accuracy:.4f}")
                            st.balloons()
                            
                        else:
                            st.error("❌ Training failed!")
                            
                    except Exception as e:
                        st.error(f"❌ Training error: {e}")
        
        # Training results
        if st.session_state.training_history is not None:
            st.markdown("### 📈 Training Results")
            
            # Create and display training chart
            training_fig = create_training_progress_chart(st.session_state.training_history)
            if training_fig:
                st.plotly_chart(training_fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 📈 Model Analytics")
        
        if st.session_state.model is None:
            st.warning("⚠️ Train a model first to see analytics.")
        else:
            # Model performance metrics
            col1, col2, col3 = st.columns(3)
            
            try:
                accuracy, cm, report = evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test)
                
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("🏗️ Model Type", st.session_state.model_type.upper())
                with col3:
                    st.metric("📊 Parameters", f"{st.session_state.model.count_params():,}")
                
                # Confusion Matrix
                st.markdown("### 🔄 Confusion Matrix")
                cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], 
                                   index=['Actual Negative', 'Actual Positive'])
                
                fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", 
                                  color_continuous_scale='Blues')
                fig_cm.update_layout(title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Classification Report
                st.markdown("### 📋 Classification Report")
                st.text(report)
                
            except Exception as e:
                st.error(f"Error generating analytics: {e}")
    
    with tab4:
        st.markdown("## ⚙️ Settings & Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Configuration")
            st.code(f"""
Dataset Path: {BASE_PATH}
GloVe Path: {GLOVE_FILE_PATH}
Max Words: {MAX_WORDS}
Max Length: {MAX_LEN}
Embedding Dim: {EMBEDDING_DIM}
            """)
            
            st.markdown("### 💾 Model Management")
            if st.button("📥 Save Current Model"):
                if st.session_state.model is not None:
                    st.info("Model saving functionality would be implemented here")
                else:
                    st.warning("No model to save")
            
            if st.button("📂 Load Saved Model"):
                st.info("Model loading functionality would be implemented here")
        
        with col2:
            st.markdown("### ℹ️ About")
            st.markdown("""
            **Sentiment Analysis Suite** is a comprehensive tool for:
            
            - 🔮 **Real-time Prediction**: Analyze text sentiment instantly
            - 🎓 **Model Training**: Train CNN, LSTM, or hybrid models
            - 📈 **Performance Analytics**: Visualize model performance
            - ⚙️ **Easy Configuration**: User-friendly interface
            
            **Built with:**
            - Streamlit for the web interface
            - TensorFlow/Keras for deep learning
            - Plotly for interactive visualizations
            - GloVe embeddings for text representation
            """)
            
            st.markdown("### 🎨 Theme")
            if st.button("🌙 Toggle Dark Mode"):
                st.info("Dark mode toggle would be implemented here")

if __name__ == "__main__":
    main()