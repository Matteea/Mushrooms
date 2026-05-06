from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

# Caricamento modelli
modello_logistic_regression = joblib.load("progetto/modelli/logistic_regression.pkl")
modello_deep_learning = joblib.load("progetto/modelli/deep_learning_model.pkl")
modello_xgb = joblib.load("progetto/modelli/xgb_model.pkl")
preprocessor = joblib.load("progetto/modelli/preprocessor.pkl")

app = FastAPI(title="Mushrooms Prediction API")
templates = Jinja2Templates(directory="templates")  # cartella contenente index.html

class_names = {0: "Velenoso", 1: "Edibile"}



class MushroomInput(BaseModel):
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


# ── GET / → form vuoto ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request, "predictions": None, "form": None, "error": None},
    )


# ── POST /predict → predizione + re-render con risultati ─────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(
        request: Request,
        # I valori arrivano già come codici a lettera singola (es. "x", "n", "t"…)
        cap_shape: str = Form(...),
        cap_surface: str = Form(...),
        cap_color: str = Form(...),
        bruises: str = Form(...),
        odor: str = Form(...),
        gill_attachment: str = Form(...),
        gill_spacing: str = Form(...),
        gill_size: str = Form(...),
        gill_color: str = Form(...),
        stalk_shape: str = Form(...),
        stalk_root: str = Form(...),
        stalk_surface_above_ring: str = Form(...),
        stalk_surface_below_ring: str = Form(...),
        stalk_color_above_ring: str = Form(...),
        stalk_color_below_ring: str = Form(...),
        veil_type: str = Form(...),
        veil_color: str = Form(...),
        ring_number: str = Form(...),
        ring_type: str = Form(...),
        spore_print_color: str = Form(...),
        population: str = Form(...),
        habitat: str = Form(...),
):
    # Dizionario dei valori del form (per ripopolare i select dopo il submit)
    form_data = dict(
        cap_shape=cap_shape, cap_surface=cap_surface, cap_color=cap_color,
        bruises=bruises, odor=odor, gill_attachment=gill_attachment,
        gill_spacing=gill_spacing, gill_size=gill_size, gill_color=gill_color,
        stalk_shape=stalk_shape, stalk_root=stalk_root,
        stalk_surface_above_ring=stalk_surface_above_ring,
        stalk_surface_below_ring=stalk_surface_below_ring,
        stalk_color_above_ring=stalk_color_above_ring,
        stalk_color_below_ring=stalk_color_below_ring,
        veil_type=veil_type, veil_color=veil_color,
        ring_number=ring_number, ring_type=ring_type,
        spore_print_color=spore_print_color, population=population,
        habitat=habitat,
    )

    try:
        # Il DataFrame usa i nomi colonna con trattino come nel dataset originale
        mushrooms = pd.DataFrame([{
            "cap-shape": cap_shape,
            "cap-surface": cap_surface,
            "cap-color": cap_color,
            "bruises": bruises,
            "odor": odor,
            "gill-attachment": gill_attachment,
            "gill-spacing": gill_spacing,
            "gill-size": gill_size,
            "gill-color": gill_color,
            "stalk-shape": stalk_shape,
            "stalk-root": stalk_root,
            "stalk-surface-above-ring": stalk_surface_above_ring,
            "stalk-surface-below-ring": stalk_surface_below_ring,
            "stalk-color-above-ring": stalk_color_above_ring,
            "stalk-color-below-ring": stalk_color_below_ring,
            "veil-type": veil_type,
            "veil-color": veil_color,
            "ring-number": ring_number,
            "ring-type": ring_type,
            "spore-print-color": spore_print_color,
            "population": population,
            "habitat": habitat,
        }])

        mushrooms_processed = preprocessor.transform(mushrooms)

        pred_lr = modello_logistic_regression.predict(mushrooms)[0]
        pred_xgb = modello_xgb.predict(mushrooms)[0]

        pred_dl_raw = modello_deep_learning.predict(mushrooms_processed)
        pred_dl = (
            (pred_dl_raw[0][0] > 0.5).astype(int)
            if pred_dl_raw.ndim == 2
            else int(pred_dl_raw[0] > 0.5)
        )

        predictions = {
            "prediction_logistic_regression": class_names[int(pred_lr)],
            "prediction_xgb": class_names[int(pred_xgb)],
            "prediction_deep_learning": class_names[int(pred_dl)],
        }

        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "predictions": predictions, "form": form_data, "error": None},
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "predictions": None, "form": form_data, "error": str(exc)},
        )



@app.post("/predict-json")
async def predict_json(data: MushroomInput):
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

    pred_lr = modello_logistic_regression.predict(mushrooms)[0]
    pred_xgb = modello_xgb.predict(mushrooms)[0]

    pred_dl_raw = modello_deep_learning.predict(mushrooms_processed)
    pred_dl = int(pred_dl_raw[0][0] > 0.5)

    return {
        "logistic_regression": class_names[int(pred_lr)],
        "xgboost": class_names[int(pred_xgb)],
        "deep_learning": class_names[int(pred_dl)],
    }