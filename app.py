import streamlit as st
import os
import tempfile
from enhance import CMGANInference

st.set_page_config(
    page_title="Audio Enhancer",
    page_icon="🎧",
    layout="centered"
)

st.title("CMGAN Audio Enhancer")
st.markdown("""
Upload a noisy audio file, CMGAN model will clean it up. 
Trained for removing background noise and echoes from voice recordings.
""")

@st.cache_resource
def load_enhancer(path):
    return CMGANInference(onnx_path=path)

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    model_path = st.text_input("Path to ONNX Model", value="cmgan.onnx")

with col2:
    tab_upload, tab_record = st.tabs(["Upload File", "Record Audio"])
    
    with tab_upload:
        uploaded_file = st.file_uploader("Upload Noisy Audio", type=['wav'])
        
    with tab_record:
        recorded_file = st.audio_input("Record Noisy Audio")

audio_data = recorded_file if recorded_file else uploaded_file

if audio_data is not None:
    st.markdown("### Original Audio")
    st.audio(audio_data)

    if st.button("Enhance Audio", type="primary"):
        if not os.path.exists(model_path):
            st.error(f"Model not found at `{model_path}`. Please check the path and try again.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
                tmp_in.write(audio_data.getvalue())
                temp_in_path = tmp_in.name
                
            temp_out_path = temp_in_path.replace(".wav", "_enhanced.wav")
            
            with st.spinner("CMGAN is cleaning your audio... This might take a moment depending on the file length."):
                try:
                    enhancer = load_enhancer(model_path)
                    enhancer.process_file(temp_in_path, temp_out_path)
                    
                    st.success("Enhancement complete!")
                    
                    with open(temp_out_path, "rb") as file:
                        audio_bytes = file.read()
                    
                    st.markdown("### Enhanced Audio")
                    st.audio(audio_bytes, format='audio/wav')

                    download_name = f"enhanced_{audio_data.name}" if hasattr(audio_data, "name") else "enhanced_recording.wav"
                    
                    st.download_button(
                        label="Download Enhanced Audio",
                        data=audio_bytes,
                        file_name=download_name,
                        mime="audio/wav"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    
                finally:
                    if os.path.exists(temp_in_path):
                        os.remove(temp_in_path)
                    if os.path.exists(temp_out_path):
                        os.remove(temp_out_path)
