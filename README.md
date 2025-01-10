# Neural-Network-Handwritten-Digit-Recognition


# Import necessary libraries
import os  # For file operations
import cv2  # For image processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
import tensorflow as tf  # For building and training machine learning models

# Load the MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the training and testing data (scale pixel values to the range 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Define the structure of the neural network model
model = tf.keras.models.Sequential()

# Input layer: Flatten the 28x28 images into a 1D array of 784 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Hidden layers: Two dense layers with 128 neurons and ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output layer: 10 neurons (for digits 0-9) with softmax activation for classification
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model: Specify optimizer, loss function, and metrics to track
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model on the training data for 3 epochs
model.fit(x_train, y_train, epochs=3)

# Save the trained model to a file
model.save('handwritten.model.keras')

# Load the trained model from the saved file
model = tf.keras.models.load_model('handwritten.model.keras')

# Initialize a counter for checking images in the 'digits' directory
image_number = 1

# Process and predict handwritten digits in the 'digits' directory
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # Read the image file
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]  # Read only the first channel
        img = np.invert(np.array([img]))  # Invert colors and convert to a NumPy array
        
        # Make a prediction using the model
        prediction = model.predict(img)
        
        # Display the prediction and show the image
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        # Handle errors during image processing or prediction
        print(f"Error with digit{image_number}.png: {e}")
    finally:
        # Increment the image counter
        image_number += 1

# Print a message when there are no more images to process
print(f"Checking for file: digits/digit{image_number}.png")
print("Program finished running.")

