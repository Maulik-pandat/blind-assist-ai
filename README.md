#  Blind Assist AI

An AI-powered assistive system designed to help visually impaired people navigate safely using real-time object detection, voice alerts, and text reading.

---

##  Project Overview

Blind Assist AI uses computer vision and deep learning to detect objects in real-time and alert the user through voice feedback.
It also estimates distance and provides direction guidance (left, right, center), helping users understand their surroundings better.

---

##  Key Features

* 🔍 Real-time object detection using YOLOv8
* 📏 Distance estimation using bounding box calculations
* 🔊 Voice alerts (Text-to-Speech system)
* 📖 OCR text reading using Tesseract
* ⚠️ Danger detection system (HIGH / NORMAL priority)
* 📊 Detection logging in CSV format
* 📈 Analytics dashboard with visual graphs

---

##  Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Tesseract OCR
* Pandas
* Matplotlib

---

##  Model Training

The system uses a pre-trained YOLOv8 model and applies transfer learning by fine-tuning it on a custom dataset collected using Roboflow.

This approach reduces training time and improves detection accuracy.

---

##  How It Works

1. Camera captures live video feed
2. YOLOv8 detects objects in the frame
3. Distance is estimated using object width
4. System determines object direction (left / center / right)
5. Voice alert is generated for the user
6. All detections are stored in a CSV file
7. Analytics script generates graphs and insights

---

##  Project Structure

blind-assist-ai/
│── train.py                # Model training
│── blind_assist_v7.py     # Main detection system
│── analytics.py           # Analytics dashboard
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── outputs/               # Screenshots and results

---

##  How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train model (optional)

python train.py

### 3. Run main system

python blind_assist_v7.py

### 4. Run analytics dashboard

python analytics.py

---

##  Sample Outputs

![Phone Detection](https://github.com/Maulik-pandat/blind-assist-ai/blob/main/outputs/Screenshot%202026-04-24%20005008.png)
![Bottle Detection](outputs/bottle.png)
![Analytics Dashboard](outputs/analytics_dashboard.png)

---

##  Output Files

* detection_log.csv → stores all detection data
* analytics_dashboard.png → generated graphs

---

##  Notes

* Model file (.pt) is not included due to large size
* detection_log.csv is generated automatically during runtime
* Works best in good lighting conditions

---

## Use Cases

* Assist visually impaired individuals
* Real-time obstacle detection
* Indoor and outdoor navigation support

---

##  Author

Maulik Bhardwaj

---

##  Future Improvements

* Mobile app integration
* GPS-based navigation system
* Wearable device support
* Improved distance accuracy using depth estimation

---
