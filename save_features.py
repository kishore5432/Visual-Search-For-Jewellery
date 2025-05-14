import pickle
from model_utils import extract_features

training_path = "static/uploads/training"

features, paths = extract_features(training_path)

with open("features.pkl", "wb") as f:
    pickle.dump((features, paths), f)

print("✅ Features extracted and saved successfully!")
