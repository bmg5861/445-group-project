import concurrent.futures
import tensorflow as tf
import numpy as np
import urllib
from PIL import Image
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split

CALLRANGE = 1
# Function to load and preprocess image from URL
def load_and_preprocess_image(url_label):
    url, label = url_label
    response = urllib.request.urlopen(url)
    img_data = response.read()
    img = Image.open(BytesIO(img_data))
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize((64, 64))  # Resize image
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img, label

def add_img_urls(api_url, list):
    r = requests.get(api_url)
    j = r.json()
    for entry in j:
        if entry["waterfall"]:
            e = {}
            e["waterfall"]=entry["waterfall"]
            e["waterfall_status"]=entry["waterfall_status"]
            list.append(e)

waterfalls = []
for calls in range(0, CALLRANGE):
    add_img_urls("https://network.satnogs.org/api/observations/?id=&status=&ground_station=2528&start=&end=&satellite__norad_cat_id=&transmitter_uuid=&transmitter_mode=&transmitter_type=&waterfall_status=1&vetted_status=&vetted_user=&observer=&observation_id=", waterfalls)
    add_img_urls("https://network.satnogs.org/api/observations/?id=&status=&ground_station=2528&start=&end=&satellite__norad_cat_id=&transmitter_uuid=&transmitter_mode=&transmitter_type=&waterfall_status=0&vetted_status=&vetted_user=&observer=&observation_id=", waterfalls)

image_data = []

for waterfall in waterfalls:
    status = waterfall.get("waterfall_status")
    if (status == "with-signal"):
        e = (waterfall.get("waterfall"), 1)
        image_data.append(e)
    else:
        e = (waterfall.get("waterfall"), 0)
        image_data.append(e)

# Load images from URLs and preprocess them in parallel
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
        tf.keras.layers.Flatten(input_shape=(64, 64)),
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
model.fit(train_images, train_labels, epochs=10)

# Evaluating results
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save("waterfall")
