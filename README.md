# YOLOv12-Based Multi-Feature Detection Framework with Voice Assistant for Enhanced Mobility and Independence of Visually Impaired Persons

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv12](https://img.shields.io/badge/YOLOv12-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE Access](https://img.shields.io/badge/Submitted-IEEE%20Access-blue.svg)](https://ieeeaccess.ieee.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-green.svg)]()

<div align="center">
  <img src="https://img.shields.io/badge/Real--Time-Detection-orange" alt="Real-Time">
  <img src="https://img.shields.io/badge/Voice-Assistant-blue" alt="Voice Assistant">
  <img src="https://img.shields.io/badge/Accessibility-Focus-brightgreen" alt="Accessibility">
</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Models](#-models)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Contributors](#-contributors)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This repository contains the implementation of a **real-time multi-feature detection framework** designed to assist visually impaired persons in achieving enhanced mobility and independence. The system integrates **YOLOv12 object detection** with **multi-modal voice feedback**, providing comprehensive environmental awareness through:

- 🚗 **Object Detection** - Real-time identification of common objects (Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, Wall)
- 💵 **Currency Recognition** - Bangladeshi currency note detection (1 Tk to 1000 Tk)
- 🚶 **Footpath Safety Assessment** - Sidewalk occupancy detection (Free, Occupied, Partially Occupied, Unsafe)
- 👤 **Face Recognition** - Known/Unknown person identification
- 📖 **Optical Character Recognition (OCR)** - Text detection with Bangla voice synthesis

### 🌟 Innovation Highlights

- **Multi-Task Detection Framework**: Three specialized YOLOv12 models running concurrently
- **Culturally Adapted**: Designed for Bangladeshi context (currency, language)
- **Real-Time Performance**: Optimized for resource-constrained devices
- **Audio-Based Interface**: Pre-recorded audio + dynamic Bangla text-to-speech
- **User-Centric Design**: Interactive mode switching for personalized assistance

---

## 🚀 Key Features

### Detection Modules

| Module | Classes | Audio Feedback | Purpose |
|--------|---------|----------------|---------|
| **Object Detection** | 9 classes | ✅ Pre-recorded | Environmental awareness |
| **Currency Detection** | 9 denominations | ✅ Pre-recorded | Financial independence |
| **Footpath Detection** | 4 conditions | ✅ Pre-recorded | Safe navigation |
| **Face Recognition** | Known/Unknown | ✅ Pre-recorded | Social interaction |
| **OCR** | English text | ✅ Generated speech | Information access |

### Technical Features

- ⚡ **Real-time Processing**: 15-30 FPS on standard hardware
- 🎙️ **Multimodal Audio**: MP3 playback + gTTS synthesis
- 🔄 **Dynamic Mode Switching**: Toggle detection modules on-the-fly
- 🛡️ **Smart Cooldown**: Prevents audio spam (configurable)
- 📊 **Performance Optimized**: Frame-based detection scheduling
- 🔌 **Flexible Input**: Webcam or video file support

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Video Stream (Webcam/File)            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Frame Preprocessing  │
                    │  - BGR Normalization  │
                    │  - 8-bit Conversion   │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  YOLOv12 #1   │   │  YOLOv12 #2   │   │  YOLOv12 #3   │
    │               │   │               │   │               │
    │   Object      │   │   Currency    │   │   Footpath    │
    │   Detection   │   │   Detection   │   │   Detection   │
    │               │   │               │   │               │
    │  (9 classes)  │   │ (9 classes)   │   │ (4 classes)   │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │ Face          │   │  OCR Module   │   │ Audio Lookup  │
    │ Recognition   │   │  (Tesseract)  │   │   Engine      │
    │ (face_recog.) │   │               │   │               │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Audio Output Manager │
                    │  - MP3 Playback       │
                    │  - gTTS Synthesis     │
                    │  - Cooldown Control   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Speaker/Headphone   │
                    └───────────────────────┘
```

---

## 📊 Datasets

### 1. Object Detection Dataset
- **Classes**: Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, Wall (9 classes)
- **Format**: YOLO annotation format
- **Purpose**: Environmental awareness for navigation

### 2. Bangladeshi Currency Dataset
- **Classes**: 1 Tk, 2 Tk, 5 Tk, 10 Tk, 20 Tk, 50 Tk, 100 Tk, 500 Tk, 1000 Tk (9 classes)
- **Format**: YOLO annotation format
- **Context**: Bangladesh-specific currency recognition

### 3. Footpath Safety Dataset
- **Classes**: Free for use, Fully Occupied, Not safe for use, Partially Occupied (4 classes)
- **Format**: YOLO annotation format
- **Purpose**: Navigation safety assessment

### 4. Face Recognition Database
- **Storage**: `Known_unknown_detection/known_faces_folder/`
- **Format**: JPG/PNG images
- **Method**: face_recognition library (dlib-based encodings)

---

## 🤖 Models

### YOLOv12 Architecture

All three detection modules utilize **YOLOv12n (nano)** for optimal speed-accuracy tradeoff:

| Model | Input Size | Parameters | Inference Speed | mAP@0.5 |
|-------|-----------|------------|-----------------|---------|
| Object Detection | 320×320 | ~3M | ~35 FPS | TBD |
| Currency Detection | 320×320 | ~3M | ~35 FPS | TBD |
| Footpath Detection | 320×320 | ~3M | ~35 FPS | TBD |

**Model Files**: Located in respective `Save Model/best.pt` directories

### Training Configuration

```python
# Model hyperparameters
imgsz: 320          # Input image size
conf: 0.35          # Confidence threshold
max_det: 5          # Maximum detections per frame
batch: 16           # Training batch size (typical)
epochs: 100         # Training epochs
optimizer: 'Adam'   # Optimization algorithm
```

---

## 💻 Installation

### Prerequisites

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **CUDA**: (Optional) CUDA 11.7+ for GPU acceleration
- **Tesseract OCR**: Required for text detection

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/object-text-detection-for-visually-impaired.git
cd object-text-detection-for-visually-impaired
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n object python=3.10
conda activate object

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

- Run the simple face detector (Known/Unknown):

```powershell
python Known_unknown_detection\known_unknown_detection.py
```

Customizing model paths
-----------------------
Edit `app.py` and change the path passed to `YOLO()` to the actual location of your `.pt` model file in the repository (or an absolute path). Example:

```python
# inside app.py
model = YOLO('Footpath Detection/best.pt')
```

Object-text-detection-for-visually-impaired