import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import requests

st.set_page_config(layout="wide", page_title="Gemini Vision Demo")

@st.cache_resource
def load_model():
    return genai.GenerativeModel('gemini-1.5-flash-001')

def process_image(image_bytes, model, prompt):
    img = Image.open(io.BytesIO(image_bytes))
    response = model.generate_content([prompt, img])
    return response.text

# Define color scheme
primary_color = "#4285F4"
secondary_color = "#34A853"
text_color = "#3C4043"

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f1f1;
        border-radius: 4px;
        color: #333;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: """ + primary_color + """;
        color: white;
    }
    h1, h2, h3 {
        color: """ + text_color + """;
    }
    .stButton>button {
        background-color: """ + secondary_color + """;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Gemini Vision Demo")

tabs = st.tabs(["Configuration", "Image Analysis"])

with tabs[0]:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key configured successfully!")
    else:
        st.warning("Please enter your Google API Key to proceed.")

with tabs[1]:
    st.header("Image Analysis")
    
    if not api_key:
        st.warning("Please configure your API Key in the Configuration tab before proceeding.")
    else:
        model = load_model()
        
        uploaded_file = st.file_uploader("Choose an image for analysis", type=["jpg", "jpeg", "png"])
        prompt = st.text_area("Enter your prompt:", "Write a short, engaging blog post based on this picture. It should include a description of the image and any relevant observations.")
        
        if uploaded_file is not None and prompt:
            image_bytes = uploaded_file.getvalue()
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    result = process_image(image_bytes, model, prompt)
                
                st.subheader("Analysis Result")
                st.write(result)
                
                st.subheader("Uploaded Image")
                st.image(image_bytes, use_column_width=True)
        elif uploaded_file is None:
            st.info("Please upload an image to analyze.")
        elif not prompt:
            st.info("Please enter a prompt for the analysis.")