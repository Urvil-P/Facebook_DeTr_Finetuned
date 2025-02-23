import streamlit as st
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont

# Load model and processor
@st.cache_resource()
def load_model():
    model_path = "detr_finetuned_on_Cars.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    num_labels = 6
    id2label = {
        0: "vehicles-and-traffic-signals",
        1: "bus",
        2: "car",
        3: "person",
        4: "traffic signal",
        5: "truck"
    }
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.config.id2label = id2label
    return model, processor, device

# Object detection function
def detect_objects(model, processor, image, device):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = [(image.size[1], image.size[0])]
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.45)[0]
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detected_objects = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label.get(label.item(), f"Unknown_{label.item()}")
        detected_objects.append(label_name)
        x_min, y_min, x_max, y_max = box.tolist()
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), f"{label_name} ({score.item():.2f})", fill="red", font=font)
    
    return image, detected_objects
   
# Streamlit UI
st.title("DeTr Cars Detection App")
model, processor, device = load_model()

uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)  
        
        with col1:
            st.image(image, caption="Uploaded Image", width=200)  
        
        processed_image, detected_objects = detect_objects(model, processor, image, device)
        
        with col2:
            st.image(processed_image, caption="Detected Objects", width=500)  
        
        st.write("---")