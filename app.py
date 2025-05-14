from flask import Flask, render_template, request
from model_utils import model
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load saved features
with open("features.pkl", "rb") as f:
    feature_list, image_paths = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join('static', 'uploads', 'test', uploaded_file.filename)
        uploaded_file.save(file_path)

        # Extract feature
        img = image.load_img(file_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        query_feature = model.predict(img_array, verbose=0)[0]

        # Compare with saved features
        similarities = np.linalg.norm(feature_list - query_feature, axis=1)
        closest_idx = np.argmin(similarities)
        result_path = image_paths[closest_idx]
        result_class = result_path.split(os.sep)[-2]  # Class name

        return render_template('index.html', query_path=file_path, result_path=result_path, result_class=result_class)

    return render_template('index.html', error="Please upload an image.")

if __name__ == '__main__':
    app.run(debug=True)
