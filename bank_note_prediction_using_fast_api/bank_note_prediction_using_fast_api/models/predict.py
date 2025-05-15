import pickle
import numpy as np
from pathlib import Path

# load model
def load_model():
    model_path = Path("models/classifier.pkl")
    # open model
    with open(model_path, 'rb') as pkl_file:
        model = pickle.load(pkl_file)

    return model

# prediction
def predict_banknote(variance: float, skewness: float, curtosis: float, entrypy: float):
    model = load_model()
    prediction = model.predict([[variance, skewness, curtosis, entrypy]])

    return "Fake note" if prediction[0] > 0.5 else "It is a bank note"
