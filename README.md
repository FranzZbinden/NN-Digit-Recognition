# Neural-Network-Handwritten-Digit-Recognition

https://github.com/user-attachments/assets/f7fd4e78-5cda-4e91-a019-e50df5754d8d

This is a project I worked on to recognize handwritten digits using a neural network. It uses the MNIST dataset for training and testing. The model was built and trained with TensorFlow/Keras, and OpenCV is used to process images for predictions.

---

## What It Does
- Trains a neural network to recognize handwritten digits (0-9).
- Saves the trained model so it can be used later.
- Allows predictions on custom images of digits stored in a folder "digits".
- Shows the predicted digit along with the input image.

---

## Adding Handwritten Digit Images
To make predictions, the program processes custom handwritten digit images. Here are the requirements for the images:

1. **Image Dimensions**: The images must be **28x28 pixels**.
2. **Color Mode**: The images should be in **grayscale** format.
3. **File Format**: Save the images in **PNG** format.
4. **Directory**: Place the images in the `digits/` folder. For example:
   - `digits/digit1.png`
   - `digits/digit2.png`

---

## Tools and Libraries
- Python 3.10 [Why Version 3.10?](#why-python-310)
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

---

### Why Python 3.10?
This project was developed using **Python 3.10**, because TensorFlow supports Python versions ranging from **3.7 to 3.11**. Using Python 3.10 ensures full compatibility with TensorFlow and other libraries.

---

## Installing Libraries
To install the libraries, run:
```bash
pip install tensorflow numpy opencv-python matplotlib
