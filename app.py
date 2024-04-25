from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import os
import tensorflow as tf

app = Flask(__name__)

# Ensure static directory exists
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)

# Correct path for the model
model = tf.saved_model.load(os.path.join(app.root_path, 'model/model89/waterfall_best_model'))
class_names = ['Without Signal', 'With Signal']

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            image_path = os.path.join('static', image_file.filename)
            image_file.save(os.path.join(app.root_path, image_path))
            result = classify_image(os.path.join(app.root_path, image_path))
            return render_template('index.html', prediction=result, img_path=image_path)
    return render_template('index.html', prediction=None, img_path=None)

def classify_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize((64, 64))  # Resize image
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = img.astype(np.float32)  # Cast to float32
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Check model's expected input shape
    print("Model expected input:", model.signatures['serving_default'].inputs[0].shape)

    # Predict the class
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)  # Create a tensor from the numpy array
    infer = model.signatures["serving_default"]
    predictions = infer(img_tensor)[list(infer.structured_outputs.keys())[0]]
    predicted_class = np.argmax(predictions.numpy())
    return class_names[predicted_class]



if __name__ == '__main__':
    app.run(debug=True)
