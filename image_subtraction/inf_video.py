import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Constants
img_height, img_width = 180, 180
class_labels = ['asthmatic', 'healthy']
threshold = 0.5  # Confidence threshold for considering a prediction

def resize_and_pad(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]), Image.Resampling.LANCZOS)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill='black')

def preprocess_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != (img_height, img_width):
            img = resize_and_pad(img, (img_height, img_width))
        img_array = np.array(img)
    return img_array

def predict_single_image(image_path,model_dir):

    model = tf.keras.models.load_model(model_dir)
    # Preprocess image
    img_array = preprocess_image(image_path)
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    # Interpret prediction
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]

    return predicted_class_index, confidence

def majority_class(predictions):
    class_counts = [0] * len(class_labels)
    for pred_class, confidence in predictions:
        if confidence >= threshold:
            class_counts[pred_class] += 1
    majority_index = np.argmax(class_counts)
    return class_labels[majority_index]

def predict_frames_in_directory(directory,model_dir):
    frame_predictions = []

    # Iterate over all image files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            pred_class, confidence = predict_single_image(image_path,model_dir)
            frame_predictions.append((pred_class, confidence))

    # Perform max pooling
    majority = majority_class(frame_predictions)

    return majority

def predict_from_two_directories(directory1, directory2, model_dir):
    frame_predictions = []

    # Iterate over all image files in the directory
    for filename in os.listdir(directory1):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory1, filename)
            pred_class, confidence = predict_single_image(image_path,model_dir)
            frame_predictions.append((pred_class, confidence))

    # Iterate over all image files in the directory
    for filename in os.listdir(directory2):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory2, filename)
            pred_class, confidence = predict_single_image(image_path,model_dir)
            frame_predictions.append((pred_class, confidence))

    # Perform max pooling
    majority = majority_class(frame_predictions)

    return majority
