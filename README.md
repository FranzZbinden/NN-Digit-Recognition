# Neural-Network-Handwritten-Digit-Recognition

https://github.com/user-attachments/assets/df06494f-9058-482e-95bd-eef450d8261f

# 🧠 NN-Digit Recognition

A Python project for handwritten digit recognition using a neural network.  
This solution uses the **MNIST** dataset to train a model (via TensorFlow / Keras) and OpenCV for preprocessing, allowing predictions on custom digit images.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Technologies & Dependencies](#-technologies--dependencies)

---

## 📖 Overview
This project implements a simple neural network to recognize handwritten digits (0–9).  
It supports:
- Training a model on the MNIST dataset  
- Saving and loading the trained model  
- Accepting custom grayscale digit images to be classified  
- Displaying predictions with visualization  

---

## ✨ Features
- End-to-end pipeline: **data → model → inference**  
- Simple architecture (easy to understand & extend)  
- Predict custom digit images (PNG, 28×28 grayscale)  
- Modular codebase  

---


## ⚙️ Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/FranzZbinden/NN-Digit-Recognition.git
   cd NN-Digit-Recognition
   
---

2. **Create & activate a virtual environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate       # for Linux / macOS
   venv\Scripts\activate          # for Windows

---

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

## 🛠 Technologies & Dependencies
- **Python 3.10**  
- [TensorFlow / Keras](https://www.tensorflow.org/)  
- [NumPy](https://numpy.org/)  
- [OpenCV](https://opencv.org/)  

All dependencies are listed in `requirements.txt`.

---

## Installing Libraries
Install all dependencies from `requirements.txt`:
```powershell
python -m pip install -r requirements.txt
```
---

## How it works internally

 ![image](https://github.com/user-attachments/assets/59431cd7-30ab-47bf-bd0b-05b4c3b4e0d7)

 ![image](https://github.com/user-attachments/assets/3654b9b0-cc2e-4b60-a524-503bd2eb51fb)


