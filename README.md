# 🔍 Image Similarity Search using MobileNetV2

This Flask web application allows users to upload an image, extracts its features using a pre-trained deep learning model (MobileNetV2), and finds the most similar image from a set of precomputed image features.

---

## 🚀 Features

* Upload an image via a web interface.
* Extract deep features using `MobileNetV2`.
* Compare uploaded image with pre-saved dataset using cosine similarity.
* Display the closest matching image and its class label.

---

## 🧠 Model

* **Architecture**: MobileNetV2 (via `tensorflow.keras.applications`)
* **Input Size**: 160x160 pixels
* **Feature Extraction**: Deep features obtained from the model are compared with pre-saved features.

---

## 🗂️ Project Structure

```
.
├── app.py                    # Main Flask application
├── model_utils.py           # Contains the loaded MobileNetV2 model
├── features.pkl             # Precomputed image features and file paths
├── templates/
│   └── index.html           # HTML template for the frontend
├── static/
│   └── uploads/test/        # Directory for uploaded test images
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <project-folder>
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have `features.pkl` and `model_utils.py` present.

4. Run the app:

   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/`.

---

## 📁 Required Files

* `features.pkl`: A pickle file containing extracted features and paths of dataset images.
* `model_utils.py`: A Python module that loads the MobileNetV2 model (must contain a `model` object).

---

## 🖼️ Usage

* Go to the home page.
* Upload an image.
* The app shows the most similar image from the dataset and its class name.

---

## 📌 Notes

* Ensure all dependencies like `TensorFlow`, `Flask`, `numpy`, and `Pillow` are installed.
* This app assumes that dataset images have been processed and their features are saved in `features.pkl`.

---

