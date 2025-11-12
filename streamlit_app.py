import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os

# Page configuration
st.set_page_config(
    page_title="Corporate Jargon Translator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, smooth styling
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Custom card styling */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07) !important;
        transition: all 0.3s ease !important;
        font-size: 16px !important;
        padding: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 6px 12px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Column containers */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    h3 {
        color: #666;
        font-weight: 500;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Model loading functions
@st.cache_resource
def load_model():
    """Load the T5 model and tokenizer from local directory"""
    try:
        model_path = "models/t5_jargon_v1"
        
        if not os.path.exists(model_path):
            return None, None, None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def translate_text(text, model, tokenizer, device):
    """Translate input text to corporate jargon"""
    input_text = f"Translate to corporate jargon: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            temperature=0.7
        )
    
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# Main app
def main():
    # Header section
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>🤖 The Dilbert-o-Matic 3000</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #888; margin-top: 0;'>Transform plain English into enterprise-grade corporate speak</h3>", unsafe_allow_html=True)
    
    st.divider()
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("⚠️ **Model not found!** Please run the training pipeline first to create the fine-tuned model in `models/t5_jargon_v1`.")
        st.info("💡 The model needs to be trained before you can translate text to corporate jargon.")
        return
    
    # Success message
    st.success(f"✅ Model loaded successfully on **{device.type.upper()}**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📝 Input")
        input_text = st.text_area(
            "Enter your plain English text:",
            height=250,
            placeholder="Type something simple like 'We need to improve our sales'...",
            label_visibility="collapsed"
        )
        
        translate_button = st.button("🚀 Translate to Jargon", use_container_width=True)
    
    with col2:
        st.markdown("### 💼 Corporate Translation")
        output_placeholder = st.empty()
    
    # Translation logic
    if translate_button and input_text.strip():
        spinner_messages = [
            "Operationalizing synergies...",
            "Recalibrating value proposition...",
            "Leveraging core competencies...",
            "Driving stakeholder alignment...",
            "Optimizing strategic frameworks...",
            "Deploying best practices..."
        ]
        
        import random
        spinner_message = random.choice(spinner_messages)
        
        with st.spinner(spinner_message):
            translated_output = translate_text(input_text, model, tokenizer, device)
        
        with output_placeholder.container():
            st.text_area(
                "Translation:",
                value=translated_output,
                height=250,
                label_visibility="collapsed"
            )
    elif translate_button:
        st.warning("⚠️ Please enter some text to translate!")
    else:
        with output_placeholder.container():
            st.text_area(
                "Translation:",
                value="Your corporate jargon will appear here...",
                height=250,
                disabled=True,
                label_visibility="collapsed"
            )
    
    st.divider()
    
    # Footer
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        ### The Dilbert-o-Matic 3000
        
        This application uses a **fine-tuned T5 transformer model** to translate plain English into corporate jargon.
        
        **Model Details:**
        - **Base Model:** T5 (Text-to-Text Transfer Transformer)
        - **Location:** `models/t5_jargon_v1`
        - **Task:** Sequence-to-sequence translation
        - **Training:** Fine-tuned on corporate jargon examples
        
        **How it works:**
        1. You input plain English text
        2. The model processes it with the prefix "Translate to corporate jargon:"
        3. The T5 model generates a jargon-filled translation
        4. Enjoy your newly synergized, strategically-aligned output! 🎯
        
        ---
        *Built with Streamlit, Transformers, and a healthy dose of corporate buzzwords.*
        """)

if __name__ == "__main__":
    main()
