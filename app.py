from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Caricamento modelli
modello_logistic_regression = joblib.load("progetto/modelli/logistic_regression.pkl")
modello_deep_learning = joblib.load("progetto/modelli/deep_learning_model.pkl")
modello_xgb = joblib.load("progetto/modelli/xgb_model.pkl")
preprocessor = joblib.load("progetto/modelli/preprocessor.pkl")


app = FastAPI(title="Mushrooms Prediction API")


class_names = {
    0: "Velenoso",
    1: "Edibile"
}


class DatiMushrooms(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str


@app.get("/")
def home():
    return {"message": "Mushroom Prediction API attiva"}


@app.post("/predict")
def predict(data: DatiMushrooms):
    mushrooms = pd.DataFrame([{
        "cap-shape": data.cap_shape,
        "cap-surface": data.cap_surface,
        "cap-color": data.cap_color,
        "bruises": data.bruises,
        "odor": data.odor,
        "gill-attachment": data.gill_attachment,
        "gill-spacing": data.gill_spacing,
        "gill-size": data.gill_size,
        "gill-color": data.gill_color,
        "stalk-shape": data.stalk_shape,
        "stalk-root": data.stalk_root,
        "stalk-surface-above-ring": data.stalk_surface_above_ring,
        "stalk-surface-below-ring": data.stalk_surface_below_ring,
        "stalk-color-above-ring": data.stalk_color_above_ring,
        "stalk-color-below-ring": data.stalk_color_below_ring,
        "veil-type": data.veil_type,
        "veil-color": data.veil_color,
        "ring-number": data.ring_number,
        "ring-type": data.ring_type,
        "spore-print-color": data.spore_print_color,
        "population": data.population,
        "habitat": data.habitat,
    }])

    mushrooms_processed = preprocessor.transform(mushrooms)

    preditcion_logistic_regression = modello_logistic_regression.predict(mushrooms)[0]
    preditcion_xgb = modello_xgb.predict(mushrooms)[0]

    prediction_deep_learning_raw = modello_deep_learning.predict(mushrooms_processed)

    prediction_deep_learning = (prediction_deep_learning_raw[0][0] > 0.5).astype(
        int) if prediction_deep_learning_raw.ndim == 2 else int(prediction_deep_learning_raw[0] > 0.5)

    return {
        "prediction_logistic_regression": class_names[int(preditcion_logistic_regression)],
        "prediction_xgb": class_names[int(preditcion_xgb)],
        "prediction_deep_learning": class_names[int(prediction_deep_learning)],
    }