# import numpy as np
# import tensorflow as tf
#
# def load_and_preprocess_image(img):
#     img = img.convert('L')  # Convert image to grayscale if not already
#     img = img.resize((64, 64))  # Resize image to (64, 64)
#     img = np.array(img) / 255.0  # Normalize pixel values
#     return img
#
# import numpy as np
# import tensorflow as tf
# import logging
#
# logger = logging.getLogger(__name__)
#
# model = None
#
# def load_model():
#     global model
#     try:
#         model = tf.saved_model.load('model/model89/waterfall_best_model')
#     except Exception as e:
#         logger.error("Failed to load model: %s", e)
#         raise e  # This will crash the app on start if model cannot be loaded, which is safer than doing it on request handling
#
# def load_and_preprocess_image(img):
#     # Your existing preprocessing steps
#     # ...
#
# def predict_radio_signal(image):
#     global model
#     try:
#         if model is None:
#             load_model()
#         infer = model.signatures['serving_default']
#         img = load_and_preprocess_image(image)
#         img = np.expand_dims(img, axis=0)
#         img = tf.convert_to_tensor(img, dtype=tf.float32)
#         prediction = infer(tf.constant(img))['output_0']
#         class_index = np.argmax(prediction)
#         return "Radio Signal Present" if class_index == 1 else "No Radio Signal"
#     except Exception as e:
#         logger.error("Prediction failed: %s", e)
#         raise e
#
# # Call load_model when starting the application
# load_model()
