import concurrent.futures
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMAGE_SIZE = (64, 64)
IMAGE_CHANNELS = 1
DATA_DIRECTORY = "" # put current working directory here

# Function to load and preprocess image from local directory
def load_and_preprocess_image(image_path_label):
    image_path, label = image_path_label
    img = Image.open(image_path)
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize(IMAGE_SIZE)  # Resize image
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img, label

# Function to add image paths and labels
def add_image_paths(directory, label, image_data):
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    image_data.extend([(image_path, label) for image_path in image_paths])

# Collect image paths and labels
image_data = []
add_image_paths(os.path.join(DATA_DIRECTORY, "withsignal"), 1, image_data)
add_image_paths(os.path.join(DATA_DIRECTORY, "withoutsignal"), 0, image_data)

# Load and preprocess images in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    images_and_labels = list(executor.map(load_and_preprocess_image, image_data))

# Convert to NumPy arrays
images, labels = zip(*images_and_labels)
images = np.array(images)
labels = np.array(labels)

# Split dataset into training and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Add batch dimension to train_images
train_images = np.expand_dims(train_images, axis=-1)

try: 
    model = tf.keras.models.load_model("waterfall")
    print("Model loaded successfully.")

except:    
    # Building the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout layer for regularization
        tf.keras.layers.Dense(64, activation='relu'),  # Adding another hidden layer
        tf.keras.layers.Dense(2)
    ])

    # Compiling the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=EPOCHS, verbose=1)

# Evaluating results
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save("waterfall")