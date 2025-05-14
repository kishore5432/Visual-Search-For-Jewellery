from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(directory):
    features = []
    image_paths = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subdir, file)
                img = image.load_img(img_path, target_size=(160, 160))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                feature = model.predict(img_array, verbose=0)[0]
                features.append(feature)
                image_paths.append(img_path)
    return features, image_paths
