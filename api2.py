import json
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

# Load model
model = tf.keras.models.load_model("my_ann_model_fix-kan.h5")
with open('fix-kan-symptoms.json', 'r') as file:
    symptoms = json.load(file)

ALL_FEATURES = symptoms['symptoms']
DISEASE_NAMES = joblib.load('disease_labels.pkl')
class SymptomInput(BaseModel):
    symptoms: list[str]

@app.post("/predict")
def predict(input_data: SymptomInput):
    input_vector = [0] * len(ALL_FEATURES)
    
    for symptom in input_data.symptoms:
        if symptom in ALL_FEATURES:
            idx = ALL_FEATURES.index(symptom)
            input_vector[idx] = 1
    
    input_array = np.array([input_vector]) 
    prediction = model.predict(input_array)
    predicted_class_idx = int(np.argmax(prediction))
    
    predicted_disease = DISEASE_NAMES[predicted_class_idx]
    confidence_percentage = float(np.max(prediction)) * 100
    
    return {
        "predicted_disease": predicted_disease,
        "confidence": f"{confidence_percentage:.2f}%"
    }
