# Neural-Network-Handwritten-Digit-Recognition

https://github.com/user-attachments/assets/78dadb71-5f56-43bd-bd0a-12c33d6475f9

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
Install all dependencies from `requirements.txt`:
```powershell
python -m pip install -r requirements.txt
# or, if Python is invoked as 'py'
py -m pip install -r requirements.txt
```
---

## Configuration (.env)

This app reads `MODEL_MODULE` from the environment. Create a `.env` file in the project root (same folder as `requirements.txt`) to override the default model module.

Example `.env`:

```
MODEL_MODULE=models.trainingModels.trainingModelTwo
```

Notes:
- The default if unset is `models.trainingModels.trainingModelTwo`.
- You can switch to the MLP with:

```
MODEL_MODULE=models.trainingModels.trainingModelOne
```

Without a `.env` file, you can also set it per session:
- PowerShell: `$env:MODEL_MODULE="models.trainingModels.trainingModelTwo"`
- Bash: `export MODEL_MODULE="models.trainingModels.trainingModelTwo"`

## How it works internally

 ![image](https://github.com/user-attachments/assets/59431cd7-30ab-47bf-bd0b-05b4c3b4e0d7)

 ![image](https://github.com/user-attachments/assets/3654b9b0-cc2e-4b60-a524-503bd2eb51fb)


