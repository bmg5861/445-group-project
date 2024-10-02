import concurrent.futures
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch

IMAGE_SIZE = (64, 64)
IMAGE_CHANNELS = 1
DATA_DIRECTORY = ""  # Set to your images' directory

# Function to load and preprocess image from local directory
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize(IMAGE_SIZE)  # Resize image
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img

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
    images_and_labels = list(executor.map(lambda x: load_and_preprocess_image(x[0]), image_data))

# Convert to NumPy arrays
images, labels = zip(*[(img, label) for (img, label) in zip(images, [label for _, label in image_data])])
images = np.array(images).reshape((-1, *IMAGE_SIZE, IMAGE_CHANNELS))
labels = np.array(labels)

# Split dataset into training and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Define the hyperparameter search space
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # Tune the number of convolutional layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(tf.keras.layers.Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(2))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Instantiate the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of hyperparameter combinations to try
    directory='hyperparameters',
    project_name='waterfall_tuning'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Search for the best hyperparameter configuration
tuner.search(train_images, train_labels, epochs=128, validation_data=(test_images, test_labels), callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Function to predict if an image contains a radio signal or not
def predict_radio_signal(image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
    prediction = best_model.predict(img)
    class_index = np.argmax(prediction)
    return "Radio Signal Present" if class_index == 1 else "No Radio Signal"

# Example usage:
result = predict_radio_signal("path_to_new_image.jpg")
print(result)

# Save the best model
best_model.save("waterfall_best_model")
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(train_images, train_labels, epochs=best_epoch, validation_data=(test_images, test_labels))
hypermodel.save("waterfall_hypermodel")
