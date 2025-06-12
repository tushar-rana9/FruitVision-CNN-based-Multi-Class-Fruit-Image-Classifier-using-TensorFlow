import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("fruit_model.keras")
 # or 'fruit_model.keras' if you saved in .keras format

# Load image to predict
img_path = "fruits-360-small/Test/Banana/banana.jpg"
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)

# Classes (Since we used only 1 class: Banana)
class_labels = ["Banana"]

print(" Starting prediction...")

print(" Predicted class:", class_labels[predicted_class_index])
