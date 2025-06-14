import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. The Core ML Components ---

# Define the model architecture
class Cnn1D(nn.Module):
    def __init__(self, num_classes=4):
        super(Cnn1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=448, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define class names
CLASS_NAMES = ['Ball Fault', 'Inner Race Fault', 'Normal', 'Outer Race Fault']

# Use Streamlit's caching to load the model and scaler only once
@st.cache_resource
def load_artifacts():
    """Loads the trained model and scaler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn1D(num_classes=4).to(device)
    
    # --- CORRECTED FILE PATHS ---
    # Provide the path relative to the repository root
    model_path = 'models/best_1d_cnn_model.pth'
    scaler_path = 'models/standard_scaler.joblib'
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler, device

# --- 2. Advanced Prediction Function ---

def predict_signal_advanced(signal_data, model, scaler, device):
    """
    Preprocesses a raw signal array of any length and returns the model's prediction.
    - Pads short signals using reflection.
    - Uses a sliding window for long signals.
    """
    WINDOW_SIZE = 2048
    STEP_SIZE = WINDOW_SIZE // 2 # 50% overlap

    # --- Handle different signal lengths ---
    if len(signal_data) < WINDOW_SIZE:
        # **MODIFIED PADDING STRATEGY**
        # Pad the signal using reflection to create a smoother, more continuous extension.
        padding_needed = WINDOW_SIZE - len(signal_data)
        padded_signal = np.pad(signal_data, (0, padding_needed), mode='reflect')
        segments = [padded_signal]
        st.info(f"Signal was shorter than {WINDOW_SIZE} samples and has been extended using reflect padding.")
    else:
        # Use a sliding window to create segments
        segments = []
        for i in range(0, len(signal_data) - WINDOW_SIZE + 1, STEP_SIZE):
            segment = signal_data[i:i + WINDOW_SIZE]
            segments.append(segment)
        if len(segments) > 1:
            st.info(f"Signal is long. Analyzing {len(segments)} overlapping windows for a more robust prediction.")

    all_probabilities = []

    for segment in segments:
        # Preprocess each segment
        segment_scaled = scaler.transform(segment.reshape(1, -1))
        segment_tensor = torch.tensor(segment_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make prediction on the segment
        with torch.no_grad():
            outputs = model(segment_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    # --- Aggregate results ---
    # Average the probabilities across all windows
    avg_probabilities = np.mean(all_probabilities, axis=0).flatten()
    
    # Determine final prediction
    final_prediction_idx = np.argmax(avg_probabilities)
    predicted_class_name = CLASS_NAMES[final_prediction_idx]
    
    # Format for output
    probabilities_dict = {CLASS_NAMES[i]: prob for i, prob in enumerate(avg_probabilities)}
    
    return predicted_class_name, probabilities_dict

# --- 3. The Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("⚙️ AI Bearing Fault Diagnosis Demo")
st.write("Upload a bearing vibration signal (as a single-column CSV) to diagnose its health. The app handles signals of any length.")

model, scaler, device = load_artifacts()

uploaded_file = st.file_uploader("Choose a signal CSV file", type="csv")

if uploaded_file is not None:
    try:
        signal_data = pd.read_csv(uploaded_file, header=None).values.flatten().astype(np.float32)
        st.success("Signal file loaded successfully!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Uploaded Vibration Signal")
            fig, ax = plt.subplots()
            ax.plot(signal_data, label="Full Signal")
            ax.set_title("Signal Waveform")
            ax.set_xlabel("Sample Number")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        predicted_class, probabilities = predict_signal_advanced(signal_data, model, scaler, device)
        
        with col2:
            st.subheader("Diagnosis Results")
            st.metric(label="Predicted Condition", value=predicted_class)
            
            st.write("Prediction Confidence:")
            
            # Create a more user-friendly bar chart with percentages
            prob_df = pd.DataFrame({
                'Fault Class': probabilities.keys(),
                'Probability': [p * 100 for p in probabilities.values()] # Convert to percentage
            })
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Probability', y='Fault Class', data=prob_df, ax=ax, orient='h')
            ax.set_xlabel('Probability (%)')
            ax.set_ylabel('')
            ax.set_xlim(0, 100)
            # Add percentage labels to the bars
            for index, value in enumerate(prob_df['Probability']):
                ax.text(value + 1, index, f'{value:.2f}%', color='black', va='center')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Awaiting file upload...")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Developed by Ahmad Nayfeh</strong></p>
    <p>🔗 <a href='https://github.com/Ahmad-Nayfeh'>GitHub</a> | 
    💼 <a href='https://www.linkedin.com/in/ahmad-nayfeh2000/'>LinkedIn</a> | 
    📧 ahmadnayfeh2000@gmail.com</p>
</div>
""", unsafe_allow_html=True)