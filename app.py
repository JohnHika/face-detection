import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("😊 Face Detection with Viola-Jones (OpenCV)")

# Instructions for the user
st.markdown("""
## 📖 How to Use This App

1. **Upload an Image**: Use the file uploader below to select an image from your device (supports JPG, JPEG, PNG formats)
2. **Adjust Parameters**: Use the sliders in the sidebar to fine-tune the face detection:
   - **Scale Factor**: Controls how much the image size is reduced at each scale (lower = more thorough detection)
   - **Min Neighbors**: Minimum number of neighbor rectangles each face needs (higher = fewer false positives)
3. **Choose Rectangle Color**: Pick your preferred color for the face detection rectangles
4. **View Results**: The app will automatically detect faces and draw rectangles around them
5. **Download Results**: Click the download button to save the processed image to your device

---
""")

# Sidebar for parameters and controls
st.sidebar.header("🎛️ Detection Parameters")

# Parameter sliders
scale_factor = st.sidebar.slider(
    "Scale Factor", 
    min_value=1.1, 
    max_value=2.0, 
    value=1.3, 
    step=0.1,
    help="How much the image size is reduced at each scale. Lower values = more thorough detection but slower processing."
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors", 
    min_value=1, 
    max_value=10, 
    value=5,
    help="Minimum number of neighbor rectangles each face needs. Higher values = fewer false positives but might miss some faces."
)

# Color picker for rectangles
rect_color = st.sidebar.color_picker(
    "Rectangle Color", 
    value="#00FF00",
    help="Choose the color for face detection rectangles"
)

# Convert hex color to BGR format for OpenCV
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
    return bgr

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image to detect faces in it"
)

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load the Haar cascade for face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces
        result_image = opencv_image.copy()
        bgr_color = hex_to_bgr(rect_color)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), bgr_color, 2)
        
        # Convert back to RGB for Streamlit display
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_image_rgb)
        
        with col2:
            st.subheader("🎯 Face Detection Results")
            st.image(result_pil, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
        
        # Display detection statistics
        st.markdown("---")
        st.subheader("📊 Detection Statistics")
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Faces Detected", len(faces))
        with col4:
            st.metric("Scale Factor", scale_factor)
        with col5:
            st.metric("Min Neighbors", min_neighbors)
        
        # Face coordinates information
        if len(faces) > 0:
            st.subheader("📍 Face Coordinates")
            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"**Face {i+1}:** Position (x={x}, y={y}), Size (width={w}, height={h})")
        
        # Download functionality
        st.markdown("---")
        st.subheader("💾 Download Processed Image")
        
        # Convert the result image to bytes for download
        import io
        
        # Create a BytesIO object
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_detection_result_{timestamp}.png"
        
        # Download button
        st.download_button(
            label="📥 Download Image with Detected Faces",
            data=img_buffer.getvalue(),
            file_name=filename,
            mime="image/png",
            help="Click to download the processed image with face detection rectangles"
        )
        
        # Success message
        if len(faces) > 0:
            st.success(f"✅ Successfully detected {len(faces)} face(s) in the image!")
        else:
            st.warning("⚠️ No faces were detected. Try adjusting the parameters in the sidebar.")
            
    except Exception as e:
        st.error(f"❌ Error loading face detection model: {str(e)}")
        st.info("Make sure OpenCV is properly installed with the required data files.")

else:
    # Display placeholder content when no image is uploaded
    st.info("👆 Please upload an image to start face detection!")
    
    # Show example of what the app can do
    st.markdown("### 🌟 Features of This App:")
    st.markdown("""
    - **Real-time face detection** using the Viola-Jones algorithm
    - **Adjustable parameters** for fine-tuning detection sensitivity
    - **Customizable rectangle colors** for better visualization
    - **Download functionality** to save processed images
    - **Detailed statistics** about detection results
    - **Face coordinate information** for each detected face
    """)
    
    st.markdown("### 🔧 Parameter Guidelines:")
    st.markdown("""
    - **Scale Factor (1.1-2.0)**: 
      - Lower values (1.1-1.2): More thorough detection, slower processing
      - Higher values (1.4-2.0): Faster processing, might miss some faces
    - **Min Neighbors (1-10)**:
      - Lower values (1-3): More detections, higher chance of false positives
      - Higher values (5-10): Fewer false positives, might miss some faces
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and OpenCV • Face Detection using Viola-Jones Algorithm*")
