import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPalette, QPainterPath, QBrush, QLinearGradient
from PyQt5.QtCore import Qt, QPoint, QTimer
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("waterfall_best_model")

# Define the class names
class_names = ['withoutsignal', 'withsignal']

def classify_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize((64, 64))  # Resize image
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    
    return class_names[predicted_class]

class WavyBackground(QWidget):
    def __init__(self):
        super().__init__()
        self.amplitude = 20
        self.frequency = 0.1
        self.phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateWave)
        self.timer.start(50)  # Update every 50 ms

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.moveTo(0, self.height())
        for x in range(self.width() + 1):
            y = self.amplitude * np.sin(self.frequency * x + self.phase) + self.height() / 2
            path.lineTo(x, y)
        path.lineTo(self.width(), self.height())
        
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, Qt.darkBlue)
        gradient.setColorAt(1, Qt.blue)
        painter.setBrush(QBrush(gradient))
        painter.drawPath(path)

    def updateWave(self):
        self.phase += 0.1
        self.update()

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 400, 400)
        
        self.background = WavyBackground()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; color: black;")

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setStyleSheet("background-color: darkgray; color: white;")

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.upload_button)
        vbox.addWidget(self.result_label)

        self.setLayout(vbox)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            result = classify_image(file_path)
            self.result_label.setText(f"Predicted Class: {result}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    ex.show()
    sys.exit(app.exec_())
