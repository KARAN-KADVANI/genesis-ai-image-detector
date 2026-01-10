import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI vs Real Detector", page_icon="🛡️", layout="centered")

# --- MODEL LOADING ---
# We use @st.cache_resource so the model stays in memory and doesn't reload on every click
@st.cache_resource
def load_trained_model():
    # Use CPU for deployment as most free hosting (like Streamlit Cloud) doesn't have GPUs
    device = torch.device("cpu") 
    
    # Recreate the exact architecture from your code
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    
    # Load your specific .pth file
    # Ensure "phase3_effnet_final.pth" is in the same folder as this app.py
    model.load_state_dict(torch.load("phase3_effnet_final.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Initialize model and device
model, device = load_trained_model()

# --- IMAGE TRANSFORMS ---
# Using your exact transformation pipeline
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- USER INTERFACE ---
st.title("🛡️ AI vs Real Image Detector")
st.write("This tool uses a fine-tuned EfficientNet-B0 to identify AI-generated patterns.")

uploaded_file = st.file_uploader("Upload an image to scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 2. Progress Bar for UX
    with st.spinner('Analyzing pixels for AI artifacts...'):
        # 3. Preprocess the image
        x = tfm(image).unsqueeze(0).to(device)

        # 4. Predict
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(1).item()
            confidence = probs[0][pred].item()

        classes = ["AI", "Real"]
        result = classes[pred]

    # 5. Display Final Result
    st.divider()
    if result == "Real":
        st.success(f"### Prediction: This image looks **REAL**")
    else:
        st.error(f"### Prediction: This image is likely **AI-GENERATED**")
        
    st.write(f"**Confidence Score:** {round(confidence * 100, 2)}%")
    
    # Optional: Progress bar visual for confidence
    st.progress(confidence)

else:
    st.info("Please upload an image file (JPG, PNG) to begin the detection.")