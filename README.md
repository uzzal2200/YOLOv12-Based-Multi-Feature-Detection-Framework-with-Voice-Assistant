# A YOLOv12–ViT Hybrid-Based Multi-Feature Detection Framework with Voice Assistant 
## for Enhanced Mobility and Independence of Visually Impaired Persons

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![YOLOv12](https://img.shields.io/badge/YOLOv12-Latest-green.svg?style=flat-square&logo=opencv)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)]()

**Real-Time Multi-Modal Assistive Technology for Environmental Awareness**

[📚 Features](#-key-features) • [⚡ Quick Start](#-quick-start) • [💾 Installation](#-installation--environment-setup) • [📊 Datasets](#-datasets) • [📖 Citation](#-citation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Models](#-models)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🎯 Overview

This repository presents a **real-time multi-feature detection framework** engineered to enhance mobility and foster independence for visually impaired individuals. The system orchestrates **three specialized YOLOv12 object detection models** with an **intelligent multimodal voice feedback system**, delivering comprehensive environmental awareness including:

- 🚗 **Object Detection** - Real-time identification of environmental objects
- 💵 **Currency Recognition** - Bangladeshi currency note denomination detection
- 🚶 **Footpath Safety Assessment** - Sidewalk occupancy and safety evaluation
- 👤 **Face Recognition** - Known/Unknown person identification
- 📖 **Optical Character Recognition** - Text detection with synthesized speech feedback

### 🌟 Key Innovation Attributes

✨ **Multi-Task Learning Architecture**: Three concurrent YOLOv12 models optimized for speed and accuracy
🌏 **Culturally Contextualized**: Designed specifically for Bangladeshi currency and language support
⚡ **Real-Time Performance**: 15-30 FPS optimized for real-world deployment
🎙️ **Multimodal Feedback**: Hybrid pre-recorded + dynamic text-to-speech interface
♿ **Accessibility-Centric**: User-friendly interactive mode switching

---

## 🚀 Key Features

### Detection Capabilities

| Module | Detection Classes | Audio Output | Use Case |
|--------|------------------|--------------|----------|
| **Object Detection** | Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, Wall (9 classes) | Pre-recorded MP3 | Environmental awareness |
| **Currency Detection** | 1Tk - 1000Tk denominations (9 classes) | Pre-recorded MP3 | Financial independence |
| **Footpath Safety** | Free/Occupied/Unsafe/Partial (4 classes) | Pre-recorded MP3 | Safe navigation |
| **Face Recognition** | Known/Unknown persons | Pre-recorded MP3 | Social interaction |
| **OCR Detection** | English text extraction | Dynamic Bangla gTTS | Information access |


---


## 📊 Datasets

### Dataset 1: Custom Object Detection
- **Source**: [Kaggle - Custom Object Detection Dataset](https://www.kaggle.com/datasets/uzzalhasan/custom-object-detection-dataset)
- **Classes**: 9 objects (Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, Wall)
- **Format**: YOLO .txt annotation format
- **Application**: General environmental object detection

### Dataset 2: Bangladeshi Currency Detection
- **Source**: [Kaggle - BD Currency Dataset](https://www.kaggle.com/datasets/uzzalhasan/bd-currency)
- **Classes**: 10 denominations (1Tk, 2Tk, 5Tk, 10Tk, 20Tk, 50Tk, 100Tk, 200Tk, 500Tk, 1000Tk)
- **Format**: YOLO .txt annotation format
- **Application**: Currency denomination recognition for financial transactions

### Dataset 3: Footpath Detection
- **Source**: [Kaggle - Footpath Detection Dataset](https://www.kaggle.com/datasets/uzzalhasan/footpath-detection)
- **Classes**: 4 conditions (Free for use, Fully Occupied, Not safe for use, Partially Occupied)
- **Format**: YOLO .txt annotation format
- **Application**: Sidewalk safety assessment for navigation

### Dataset 4: Face Recognition Database
- **Storage**: `Known_unknown_detection/known_faces_folder/`
- **Format**: JPG/PNG image files
- **Application**: Person identification and social interaction

---

### 6. OCR Detection Module

```bash
python "OCR detection/OCR_Bangla_english.py"
```

---

## 📁 Project Structure

```
object-text-detection-for-visually-impaired/
│
├── app.py                                    # Main real-time detection pipeline
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # MIT License
├── README.md                                 # This file
├── .gitignore                                # Git ignore file
│
├── audio/                                    # Pre-recorded audio feedback files (26 files)
│   ├── 1 tk.mp3, 2 taka.mp3, 5 tk.mp3, 10 Tk.mp3, 20 tk.mp3, 50 tk.mp3, 100 tk.mp3, 200 tk.mp3, 500 tk.mp3, 1000 tk.mp3  # Currency audio (10 files)
│   ├── Vehicle.mp3, Chair.mp3, Door.mp3, Man.mp3, Road.mp3, Stair.mp3, Table.mp3, Tree.mp3, wall.mp3  # Object detection audio (9 files)
│   ├── free for use.mp3, Fully Occupied .mp3, Partially Occupied .mp3, Not safe for use.mp3  # Footpath audio (4 files)
│   ├── Known Face Uzzal .mp3, Unknown Face.mp3  # Face recognition audio (2 files)
│   
│
├── Object detection Custom dataset/
│   ├── custom_object_detection_with_yolov12n_pt.ipynb  # Training notebook
│   └── Save Model/
│       ├── best.pt                          # Best trained YOLOv12 model
│       └── last.pt                          # Last checkpoint
│
├── Bangladesh Currency Detection/
│   ├── Bangladeshi_Currency_detection_with_yolov12n_pt.ipynb  # Training notebook
│   └── Save Model/
│       └── best.pt                          # Trained currency detection model
│
├── Footpath Detection/
│   ├── Footpath_detection_yolov12n_pt.ipynb  # Training notebook
│   └── Save Model/
│       └── best.pt                          # Trained footpath detection model
│
├── Known_unknown_detection/
│   ├── known_unknown_detection.py           # Face recognition detection script
│   ├── evaluation_metrices.py               # Evaluation metrics for face detection
│   ├── known_faces_folder/                  # Database of known person face images
│   └── .venv/                               # Virtual environment
│
├── OCR detection/
│   ├── OCR_Bangla_english.py                # OCR text detection script
│   ├── evaluation_metrices.py               # Evaluation metrics for OCR
│   └── __pycache__/                         # Python cache files
│
├── .git/                                     # Git version control repository
│
└── YOLOv12_Based_Multi_Feature_Detection...pdf  # Research paper PDF
```

---


## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv12 framework
- **dlib** community for face recognition
- **Tesseract-OCR** project for text detection
- **Kaggle** for dataset resources
- **Open source community** for PyTorch, OpenCV, and other dependencies

---

## 📞 Support & Contribution

For issues, feature requests, or contributions, please open an issue or submit a pull request on GitHub.

**Last Updated**: January 27, 2026
