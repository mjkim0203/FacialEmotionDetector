# Facial Emotion Detector

A Flask-based real-time facial emotion detection app using OpenCV and a pre-trained Keras/TensorFlow model.

## 🚀 Features

- Real-time webcam video stream
- Face detection using Haar Cascades
- Emotion classification with CNN model (7 emotions)
- Live overlay of predictions

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for all packages.

Main dependencies:
- Flask
- OpenCV
- Keras
- TensorFlow (CPU version)
- NumPy

---

## 🖥️ Local Run (Recommended for webcam access)

```bash
# Clone the repository
git clone https://github.com/mjkim0203/FacialEmotionDetector.git
cd FacialEmotionDetector

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
