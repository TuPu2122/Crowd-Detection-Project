# PBL DEEPLEARNING
# 🚀 Real-Time Crowd Detection and Future Prediction using YOLOv8 & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This project presents a smart system for **crowd detection, analysis, and prediction** using Deep Learning and Machine Learning techniques.

It uses **YOLOv8** for real-time person detection and generates a structured dataset to analyze crowd patterns and predict future crowd density.

The system is useful for **college campuses, events, and public areas** where crowd monitoring is important.

---

## 🎯 Objectives

- Detect and count people using YOLOv8  
- Generate structured dataset (CSV format)  
- Analyze crowd patterns (time, location, day type)  
- Predict future crowd density using ML models  
- Build a scalable system for real-world applications  

---

## ✨ Features

- 🎥 Real-time crowd detection  
- 👥 Accurate people counting using bounding boxes  
- 📊 Automated CSV dataset generation  
- 📈 Data analysis and visualization  
- 🔮 Future crowd prediction using Machine Learning  
- ⚡ Optimized and scalable pipeline  

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|--------|
| Python | Core Programming |
| YOLOv8 (Ultralytics) | Object Detection |
| OpenCV | Image Processing |
| Pandas | Data Handling |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Scikit-learn | Machine Learning |
| Flask / Streamlit | Web Interface |

---

## 🧠 Workflow

1. Image Input  
2. YOLOv8 Detection  
3. Crowd Counting  
4. Data Encoding (time, location, day type)  
5. CSV Dataset Generation  
6. Data Analysis & Visualization  
7. Machine Learning Prediction  
8. Web Interface (optional)

---

## 📂 Project Structure
Crowd-Detection-Project/
│
├── data/
│ ├── raw_images/
│ ├── processed_images/
│ └── dataset.csv
│
├── models/
│ └── yolo_model.pt
│
├── src/
│ ├── detection.py
│ ├── preprocessing.py
│ ├── dataset_generator.py
│ ├── prediction_model.py
│ └── app.py
│
├── outputs/
│ ├── graphs/
│ └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
