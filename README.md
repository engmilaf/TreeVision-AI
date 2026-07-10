<h1 align="center">🌳 TreeVision AI</h1>

<p align="center">
<b>Tree Species Classification Using Teachable Machine and Google Colab</b>
</p>

<p align="center">
An Artificial Intelligence project that classifies tree species from images using Machine Learning and Computer Vision.
</p>

<hr>

<h2>📖 About the Project</h2>

<p>
TreeVision AI is an educational AI project that demonstrates how image classification can be used to recognize different tree species from images.
The model was trained using <strong>Teachable Machine</strong> and tested in <strong>Google Colab</strong> using Python.
</p>

<hr>

<h2>🎯 Project Objective</h2>

<p>
The purpose of this project is to build an AI model capable of recognizing tree species from images and to demonstrate the complete image classification workflow from training to prediction.
</p>

<hr>

<h2>🌲 Tree Classes</h2>

<ul>
<li>🌴 Palm Tree</li>
<li>🌲 Pine Tree</li>
<li>🌳 Oak Tree</li>
</ul>

<hr>

<h2>📂 Dataset</h2>

<p>
A custom dataset containing <strong>15 images</strong> was created for this project, with five images for each tree category.
</p>

<table>
<tr>
<th>Tree Species</th>
<th>Images</th>
</tr>

<tr>
<td>🌴 Palm Tree</td>
<td>5</td>
</tr>

<tr>
<td>🌲 Pine Tree</td>
<td>5</td>
</tr>

<tr>
<td>🌳 Oak Tree</td>
<td>5</td>
</tr>

</table>

<hr>

<h2>⚙️ Project Workflow</h2>

<ol>
<li>Collect tree images.</li>
<li>Upload the dataset to Teachable Machine.</li>
<li>Train and validate the model.</li>
<li>Export the trained model.</li>
<li>Load the model into Google Colab.</li>
<li>Predict the tree species using Python.</li>
</ol>

<hr>

## 💻 Google Colab Implementation

The exported model was tested in **Google Colab** using Python. The following code loads the trained model, preprocesses the input image, performs image classification, and displays the predicted tree species along with the confidence score.

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tf_keras as tk

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the trained model
model = tk.models.load_model("keras_model.h5", compile=False)

# Load the class labels
class_names = open("labels.txt", "r").readlines()

# Create an input array for the model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load the input image
image = Image.open("tree.jpg").convert("RGB")

# Resize the image to the required size
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert the image to a NumPy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Run prediction
prediction = model.predict(data)

# Get the predicted class
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Display prediction
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```

<h2>🛠 Technologies Used</h2>

<ul>
<li>🤖 Teachable Machine</li>
<li>☁️ Google Colab</li>
<li>🐍 Python</li>
<li>📦 TensorFlow / Keras</li>
<li>🧠 Machine Learning</li>
<li>👁️ Computer Vision</li>
</ul>

<hr>

<h2>📊 Results</h2>

<p>
The trained model successfully recognized the three tree species:
</p>

<ul>
<li>🌴 Palm Tree</li>
<li>🌲 Pine Tree</li>
<li>🌳 Oak Tree</li>
</ul>

<p>
The exported model also produced successful predictions in Google Colab, displaying both the predicted class and the confidence score.
</p>

<hr>

<h2>🚀 Future Improvements</h2>

<ul>
<li>Add more tree species.</li>
<li>Increase the dataset size.</li>
<li>Improve prediction accuracy.</li>
<li>Develop a web or mobile application.</li>
</ul>

<hr>

<h2>📌 Conclusion</h2>

<p>
TreeVision AI demonstrates a complete image classification workflow using Teachable Machine and Google Colab. It provides a simple and practical introduction to Artificial Intelligence, Machine Learning, and Computer Vision.
</p>

<hr>

<p align="center">
Made with ❤️ using Teachable Machine, Google Colab, and Python.
</p>
