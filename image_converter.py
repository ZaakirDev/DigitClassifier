"""
THIS PROGRAM IS USED TO CONVERT THE IMAGES IN THE MNIST CSV FILES INTO REAL 28x28 BIT IMAGES 
TO BE FED INTO THE APPLICATION.
"""

import numpy as np
from PIL import Image
import pandas as pd

data_test = pd.read_csv("./mnist_test.csv")
data_test = np.array(data_test)

y_test = data_test[:, 0]  # Labels
X_test = data_test[:, 1:]  # Features (pixel values)

# Normalize pixel values
X_test = X_test / 255.0

count = 0

# Loop through all the samples in testing
while count <= len(y_test) - 1:
    # Convert each sample into an image and save it in a designated folder
    canvas = X_test[count]
    canvas_2d = (canvas.reshape(28, 28) * 255).astype(np.uint8)
    Image.fromarray(canvas_2d, mode='L').save(f"./ImageData/{count}.png")

    count += 1

