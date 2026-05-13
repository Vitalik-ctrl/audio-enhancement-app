# CMGAN Audio Enhancer

A Streamlit web application for removing background noise and echoes from audio recordings using state-of-the-art CMGAN (Conformer-based Generative Adversarial Networks) models.

## Features

- **Multiple CMGAN Models**: Choose from 6 different models including generalist and specialist variants optimized for different scenarios and languages
- **Flexible Input Methods**: Upload audio files or record directly from your microphone
- **GPU Acceleration**: Automatic GPU support with CUDA (with CPU fallback)
- **Real-time Processing**: Stream enhancement feedback with progress indicators
- **Audio Format Support**: Process WAV and WV1 (NIST format) files
- **Easy Download**: Download enhanced audio instantly after processing
- **Web Interface**: User-friendly Streamlit interface, no CLI required

## Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- ONNX Runtime
- Streamlit
- LibROSA
- SoundFile

## Installation

1. **Clone/Download the project** to your machine
   ```bash
   cd audio-enhancement-frontend
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note**: If you have an NVIDIA GPU, replace the PyTorch index in `requirements.txt` to use GPU wheels for faster processing

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Select a CMGAN Model**
   - **Generalist New** (cmgan_generalist_final.onnx): Latest generalist model with improved overall performance
   - **Generalist Old** (cmgan.onnx): Original generalist model variant
   - **English Specialist** (english_specialist.onnx): Optimized for English language speech enhancement
   - **Czech Specialist** (czech_specialist.onnx): Optimized for Czech language speech enhancement
   - **RIR Specialist** (rir_specialist.onnx): Specialized for echo and room impulse response removal
   - **Additive Specialist** (additive_specialist.onnx): Specialized for additive noise suppression 

4. **Input Audio**
   - **Upload File**: Click "Upload File" tab and select a WAV or WV1 audio file
   - **Record Audio**: Click "Record Audio" tab to capture audio directly

5. **Process Audio**
   - Click the "Enhance Audio" button
   - Wait for processing to complete
   - Review the enhanced audio

6. **Download Results**
   - Click "Download Enhanced Audio" to save the cleaned file

## Project Structure

```
audio-enhancement-frontend/
├── app.py                          # Main Streamlit application
├── enhance.py                      # CMGAN inference engine
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/                         # CMGAN model files
│   ├── cmgan.onnx                  # CMGAN model (generalist old)
│   ├── cmgan_generalist_final.onnx # CMGAN model (generalist new)
│   ├── english_specialist.onnx     # English language specialist model
│   ├── czech_specialist.onnx       # Czech language specialist model
│   ├── rir_specialist.onnx         # Room impulse response specialist model
│   └── additive_specialist.onnx    # Additive noise specialist model
└── data/                           # Sample data and results
```

## Data Folder Structure

| Path | Type | Purpose | Contents |
|------|------|---------|----------|
| `data/` | Directory | Root data folder | Contains sample audio files and results |
| `data/to_enhance/` | Directory | Input audio directory | Noisy audio files awaiting enhancement |
| `data/enhanced/` | Directory | Output results directory | Enhanced audio files organized by test set |
| `data/enhanced/enhanced_custom/` | Directory | Custom test results | Results from custom audio samples |
| `data/enhanced/enhanced_custom/SA178S01_lady_cz/` | Directory | Czech female speaker results | Enhanced audio from SA178S01_lady_cz dataset |
| `data/enhanced/enhanced_custom/si1354_gent_en/` | Directory | English male speaker results | Enhanced audio from si1354_gent_en dataset |
| `data/enhanced/enhanced_custom/si584_lady_en/` | Directory | English female speaker results | Enhanced audio from si584_lady_en dataset |

## Technical Details

### CMGAN Model
- **Architecture**: Conformer-based Generative Adversarial Networks
- **Input**: 16kHz mono audio (16-bit PCM WAV format)
- **Processing**: Processes audio in 4-second chunks for memory efficiency
- **Output**: Enhanced 16kHz mono audio (normalized to prevent clipping)
- **Inference**: ONNX Runtime with GPU acceleration support

| `Available Models

| Model Name | Filename | Best For | Focus Area |
|------------|----------|----------|-----------|
| Generalist New | cmgan_generalist_final.onnx | General-purpose enhancement | All noise types & languages |
| Generalist Old | cmgan.onnx | General-purpose enhancement | All noise types & languages |
| English Specialist | english_specialist.onnx | English speech | Language-specific optimization |
| Czech Specialist | czech_specialist.onnx | Czech speech | Language-specific optimization |
| RIR Specialist | rir_specialist.onnx | Echo removal | Room impulse response & reverb |
| Additive Specialist | additive_specialist.onnx | Noise suppression | Additive background noise |

**Recommendation**: Start with "Generalist New" for best overall results. Switch to specialist models for specific use cases or if you just want to compare them between each other.

### Audio Processing Pipeline
1. **Resampling**: Normalizes audio to 16kHz sample rate
2. **Chunking**: Splits audio into manageable chunks
3. **STFT Transform**: Converts time-domain to frequency domain
4. **Power Compression**: Applies perceptual power compression (0.3)
5. **Model Inference**: Runs CMGAN ONNX model
6. **Reconstruction**: Inverse STFT to get enhanced time-domain signal
7. **Normalization**: Prevents clipping by normalizing peak values

## Supported Audio Formats

- **WAV** (.wav): Standard PCM audio format
- **WV1** (.wv1): NIST format (automatically converted to WAV internally)

## System Requirements

### Minimum (CPU Processing)
- 4GB RAM
- Modern CPU
- ~500MB disk space

### Recommended (GPU Processing)
- NVIDIA GPU with CUDA Compute Capability 3.5+
- 8GB+ VRAM for GPU
- 8GB+ system RAM
- ~1GB disk space (for models)

## Troubleshooting

### Model file not found
- Ensure `cmgan.onnx` and `cmgan_generalist_final.onnx` are in the application root directory

### GPU not detected
- Install CUDA Toolkit and cuDNN if you have an NVIDIA GPU
- Reinstall onnxruntime: `pip install onnxruntime-gpu`

### Audio processing is slow
- Use GPU if available (see GPU installation above)
- Process shorter audio files
- Reduce chunk size in `enhance.py` (advanced)

### Audio quality issues
- Ensure input audio is below 0dB peak to avoid clipping
- Try the alternative CMGAN model
- Ensure input is 16-bit PCM format
