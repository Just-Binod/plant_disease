import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Simple custom CSS for footer only
st.markdown("""
<style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        padding: 10px;
        text-align: center;
        color: #fafafa;
        border-top: 1px solid #303030;
        z-index: 1000;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
        font-weight: 500;
    }
    .footer a:hover {
        color: #45a049;
        text-decoration: underline;
    }
    .main-content {
        margin-bottom: 60px;
    }
    /* Small enhancement for buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #45a049;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title with slight enhancement
st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf photo to check if it's **healthy** or **diseased**!")

# Sidebar (keeping your original content)
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app uses **MobileNetV3** trained on PlantVillage dataset.
    
    **Classes:**
    - 🟢 Healthy
    - 🔴 Diseased
    
    Model accuracy: **99.56%** on test data
    """)
    
    st.header("📸 Tips")
    st.warning("""
    - Use clear, well-lit photos
    - Focus on the leaf surface
    - Avoid blurry images
    """)
    
    # Added a small separator
    st.divider()
    
    # Quick guide
    st.caption("🔄 **How to use:** Upload → Click Diagnose → Get Results")

# Wrap main content for footer spacing
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    try:
        # Create model architecture (same as your notebook)
        model = models.mobilenet_v3_large(weights=None)
        
        # Replace classifier for binary classification (2 classes)
        model.classifier[3] = nn.Linear(1280, 2)  # 2 classes: healthy(0), diseased(1)
        
        # Load your trained weights
        model_path = 'plant_disease_model.pth'
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please upload your model file to the app directory")
            return None
            
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define transforms (MUST match your training transforms)
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Define class names (matching your binary mapping)
class_names = ['healthy', 'diseased']  # 0=healthy, 1=diseased

# Main app
def main():
    # Load model
    model = load_model()
    if model is None:
        return
    
    transform = get_transforms()
    
    # File uploader with better description
    uploaded_file = st.file_uploader(
        "📤 **Choose a leaf image...**", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'JPG'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Display image with caption
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📸 Uploaded Image', use_container_width=True)
        
        # Predict button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button('🔍 Diagnose Plant', type='primary', use_container_width=True):
                with st.spinner('🔬 Analyzing leaf...'):
                    # Preprocess
                    input_tensor = transform(image).unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        confidence, predicted = torch.max(probabilities, 0)
                    
                    # Show results
                    st.success("✅ Analysis Complete!")
                    
                    # Result with emoji
                    col1, col2 = st.columns(2)
                    with col1:
                        if predicted.item() == 0:
                            st.markdown("### 🟢 **HEALTHY**")
                        else:
                            st.markdown("### 🔴 **DISEASED**")
                    with col2:
                        # Create a metric for confidence
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Progress bars with better labels
                    st.subheader("📊 Detailed Analysis")
                    prob_healthy = probabilities[0].item()
                    prob_diseased = probabilities[1].item()
                    
                    # Healthy progress
                    st.markdown("**🟢 Healthy**")
                    st.progress(prob_healthy, text=f"{prob_healthy:.1%}")
                    
                    # Diseased progress
                    st.markdown("**🔴 Diseased**")
                    st.progress(prob_diseased, text=f"{prob_diseased:.1%}")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if predicted.item() == 0:
                        st.info("🌱 Your plant appears healthy! Continue regular care and monitoring.")
                    else:
                        st.warning("""
                        🍂 **Disease detected!** Consider:
                        - Isolating the affected plant
                        - Consulting a local agriculture expert
                        - Removing severely affected leaves
                        """)
                    
                    # Add timestamp
                    st.caption(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()

# Close main content div
st.markdown('</div>', unsafe_allow_html=True)

# FOOTER with your name and LinkedIn (exactly as you wanted)
st.markdown(f"""
<div class="footer">
    Developed with ❤️ by <strong>Binod Raj Pant</strong> | 
    <a href="https://www.linkedin.com/in/binod-raj-pant-303767330/" target="_blank">Connect on LinkedIn 🔗</a>
</div>
""", unsafe_allow_html=True)