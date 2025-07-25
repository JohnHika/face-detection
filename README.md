# 😊 Face Detection App with Viola-Jones Algorithm

A comprehensive Streamlit web application for real-time face detection using OpenCV's Viola-Jones algorithm.

## 🌟 Features

- **Real-time face detection** using the Viola-Jones algorithm
- **Interactive parameter adjustment** for fine-tuning detection sensitivity
- **Customizable rectangle colors** for better visualization
- **Download functionality** to save processed images
- **Detailed statistics** about detection results
- **Face coordinate information** for each detected face
- **User-friendly interface** with clear instructions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JohnHika/face-detection.git
cd face-detection
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install opencv-python streamlit numpy pillow
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎛️ How to Use

1. **Upload an Image**: Use the file uploader to select an image (JPG, JPEG, PNG)
2. **Adjust Parameters**: 
   - **Scale Factor**: Controls detection thoroughness (1.1-2.0)
   - **Min Neighbors**: Controls false positive filtering (1-10)
3. **Choose Rectangle Color**: Pick your preferred color for face detection rectangles
4. **View Results**: The app automatically detects faces and draws rectangles around them
5. **Download Results**: Save the processed image to your device

## 🔧 Parameter Guidelines

- **Scale Factor (1.1-2.0)**:
  - Lower values (1.1-1.2): More thorough detection, slower processing
  - Higher values (1.4-2.0): Faster processing, might miss some faces

- **Min Neighbors (1-10)**:
  - Lower values (1-3): More detections, higher chance of false positives
  - Higher values (5-10): Fewer false positives, might miss some faces

## 📦 Dependencies

- `streamlit` - Web app framework
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing
- `pillow` - Image processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**John Hika**

- GitHub: [@JohnHika](https://github.com/JohnHika)

## 🙏 Acknowledgments

- OpenCV team for the Haar cascade classifiers
- Streamlit team for the amazing web framework
- Viola-Jones algorithm for face detection
