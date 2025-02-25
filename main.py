import streamlit as st
import numpy as np
import io
import pydub
from df.enhance import enhance, init_df, save_audio
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import librosa
import torch
from zipfile import ZipFile
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Gurukul Audio Cleaner",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, clean CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        padding: 1em;
        background-color: #FFFFFF;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        padding: 1em 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0 0 15px 15px;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .header h1 {
        color: #FFFFFF;
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 0.1em;
        letter-spacing: -0.5px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9em;
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #F8FAFF;
    }
    
    .sidebar-file {
        background: white;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 0.5em;
        border: 1px solid #E5E9FF;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8em 1.5em;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Download button styling */
    .download-btn {
        margin: 1em 0;
        text-align: center;
    }
    
    audio {
        width: 100%;
        margin: 0.5em 0;
        border-radius: 8px;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for processed files
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
    st.session_state.zip_data = None

# Sidebar for file management
with st.sidebar:
    st.markdown("### 📁 File Manager")
    uploaded_files = st.file_uploader(
        "Upload Audio Files",
        accept_multiple_files=True,
        help="Supports WAV and MP3 formats",
        key="file_uploader"
    )
    
    if uploaded_files:
        st.markdown("---")
        st.markdown("### 📂 Uploaded Files")
        for file in uploaded_files:
            st.markdown(f"🎵 {file.name}")

# Main content area
st.markdown("""
    <div class="header">
        <h1>Gurukul Audio Cleaner</h1>
        <p>Clean audio with AI-powered noise reduction</p>
    </div>
""", unsafe_allow_html=True)

# Main processing section
if uploaded_files:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('✨ Enhance Audio', use_container_width=True):
            with st.container():
                progress = st.progress(0)
                status = st.empty()
                
                # Initialize model
                with st.spinner("Preparing AI model..."):
                    model, df_state, _ = init_df()
                
                # Process files
                with tempfile.TemporaryDirectory() as temp_dir:
                    processed_files = []
                    
                    for idx, file in enumerate(uploaded_files):
                        status.info(f"Processing {file.name}...")
                        
                        # Process audio
                        audio_array, sample_rate = librosa.load(file, sr=None)
                        audio_stream = io.BytesIO(file.getvalue())
                        audio = pydub.AudioSegment.from_file(audio_stream)
                        temp_audio_file = io.BytesIO()
                        audio.export(temp_audio_file, format="wav")
                        
                        # Clean audio
                        waveform, _ = torchaudio.load(io.BytesIO(temp_audio_file.getvalue()))
                        enhanced = enhance(model, df_state, waveform)
                        enhanced_numpy = enhanced.cpu().numpy()
                        
                        # Save processed file
                        output_filename = os.path.join(temp_dir, f"enhanced_{file.name}")
                        sf.write(output_filename, enhanced_numpy.T, sample_rate)
                        
                        # Store processed file info
                        processed_files.append({
                            'name': file.name,
                            'enhanced_audio': enhanced_numpy,
                            'sample_rate': sample_rate
                        })
                        
                        progress.progress((idx + 1) / len(uploaded_files))
                    
                    # Create zip file
                    zip_path = os.path.join(temp_dir, "enhanced_audio.zip")
                    with ZipFile(zip_path, 'w') as zip_file:
                        for file in os.listdir(temp_dir):
                            if file.startswith("enhanced_"):
                                zip_file.write(os.path.join(temp_dir, file), file)
                    
                    # Store processed files in session state
                    with open(zip_path, "rb") as f:
                        st.session_state.zip_data = f.read()
                    st.session_state.processed_files = processed_files
                    
                    status.success("✨ Enhancement complete!")

# Display processed files and download options
if st.session_state.processed_files:
    # Main download button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            "⬇️ Download All Enhanced Files",
            data=st.session_state.zip_data,
            file_name="enhanced_audio.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # Display audio previews
    st.markdown("### 🎵 Enhanced Audio Files")
    for file_info in st.session_state.processed_files:
        with st.expander(f"Preview: {file_info['name']}", expanded=False):
            st.audio(file_info['enhanced_audio'], format='audio/wav', sample_rate=file_info['sample_rate'])
            st.download_button(
                f"⬇️ Download {file_info['name']}",
                data=st.session_state.zip_data,
                file_name=f"enhanced_{file_info['name']}",
                mime="application/zip",
                use_container_width=True
            )
else:
    # Empty space instead of the drop message
    st.write("")
