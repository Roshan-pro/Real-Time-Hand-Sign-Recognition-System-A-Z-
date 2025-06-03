# Real-Time-Hand-Sign-Recognition-System-A-Z-
# 🤚 Real-Time Hand Sign Recognition (A–Z) using Deep Learning

A real-time hand sign recognition system built **from scratch**, using a custom CNN trained on the ASL alphabet dataset from Kaggle. This project detects hand gestures via webcam and classifies them into 29 classes, including all 26 alphabets, `SPACE`, `DEL`, and `NOTHING`.


---

## 🔍 Project Highlights

- 🔠 Classifies 29 ASL classes: A–Z + SPACE, DEL, NOTHING  
- 🧠 Deep Learning model trained **from scratch** (no pretrained networks)  
- 🎥 Real-time prediction from webcam feed using OpenCV  
- 🖐️ Hand detection powered by **MediaPipe**  
- 📷 Frame-by-frame live classification with prediction overlay  

---

## 📁 Dataset

- **Source**: [Kaggle - ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)  
- ~87,000 labeled images (2000+ per class)  
- Resized to 128×128 RGB images  
- Custom preprocessing pipeline applied  

---

## 🧠 Model Architecture

- **Input**: 128x128x3 RGB  
- **Layers**:  
  - 4× Conv2D → ReLU → MaxPooling → Dropout  
  - Flatten → Dense → Softmax (29 classes)  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Framework**: TensorFlow / Keras  

> Trained **without** using any pretrained models like VGG or ResNet.

---

## 🖥️ Real-Time Prediction Flow

1. Capture frame from webcam  
2. Detect hand using **MediaPipe**  
3. Crop and preprocess hand region (128×128)  
4. Predict label using trained CNN  
5. Display prediction live on webcam feed  

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hand-sign-recognition.git
cd hand-sign-recognition
```
##2. Install Dependencies
```blash
pip install -r requirements.txt
```
## Requirements
- Python 3.10+
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
# ##👨‍💻 Author
Roshan Kumar
B.Sc. Computer Science and Data Analytics
Indian Institute of Technology, Patna

📫 Email:rk1861303@gmil.com
🔗 LinkedIn:https://www.linkedin.com/in/roshan-kumar-670529317/
💻 GitHub



