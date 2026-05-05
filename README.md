# 🍄 Mushroom Edibility Classification

Progetto di classificazione binaria per predire la **commestibilità dei funghi** (velenosi vs commestibili) attraverso tecniche di Machine Learning tradizionale e Deep Learning, utilizzando il dataset **UCI Mushroom (ID 73)**.

---

## 📋 Indice

- [Panoramica del Progetto](#-panoramica-del-progetto)
- [Dataset](#-dataset)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Pipeline dei Dati](#-pipeline-dei-dati)
- [Modelli Implementati](#-modelli-implementati)
- [Risultati](#-risultati)
- [Salvataggio di Modelli e Risultati](#-salvataggio-di-modelli-e-risultati)

---

## 🔍 Panoramica del Progetto

L'obiettivo è costruire un classificatore binario in grado di determinare, a partire da caratteristiche morfologiche di un fungo, se esso sia **commestibile (edible)** o **velenoso (poisonous)**.

Il progetto confronta tre approcci distinti:

| Modello | Tipo | Libreria |
|---|---|---|
| XGBoost Classifier | Gradient Boosting | `xgboost` |
| Logistic Regression | Modello Lineare | `scikit-learn` |
| Rete Neurale | Deep Learning | `TensorFlow / Keras` |

---

## 📊 Dataset

Il dataset utilizzato è il **[UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)** (ID: 73), scaricato direttamente tramite la libreria `ucimlrepo`.

### Caratteristiche principali

- **Istanze:** 8.124 campioni
- **Feature:** 22 attributi categorici (forma del cappello, colore, odore, tipo di lamelle, ecc.)
- **Target:** `Edible` → `e` (commestibile) / `p` (velenoso)
- **Missing values:** la colonna `stalk-root` contiene valori nulli, trattati con imputazione per moda

### Feature del Dataset

Le 22 feature originali descrivono caratteristiche morfologiche del fungo, tra cui:
`cap-shape`, `cap-surface`, `cap-color`, `bruises`, `odor`, `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color`, `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring`, `veil-type`, `veil-color`, `ring-number`, `ring-type`, `spore-print-color`, `population`, `habitat`

### Encoding del Target

```
"e" (edible)   → 1
"p" (poisonous) → 0
```

---

## 📁 Struttura del Progetto

```
progetto/
│
├── main.py                  # Entry point: orchestrazione dell'intera pipeline
├── data_manager.py          # Caricamento, pulizia, preprocessing e split dei dati
├── machine_learning.py      # XGBoost Classifier e Logistic Regression
├── deep_learning.py         # Rete neurale con TensorFlow/Keras
│
├── modelli/                 # Modelli addestrati serializzati
│   ├── preprocessor.pkl
│   ├── xgb_model.pkl
│   ├── logistic_regression.pkl
│   └── deep_learning_model.pkl
│
└── risultati/               # Confusion matrix generate durante il training
    ├── XGBClassifier_confusion_matrix.png
    ├── Logistic Regressor_confusion_matrix.png
    └── Deep Learning_confusion_matrix.png
```

---

## 🔄 Pipeline dei Dati

La gestione dei dati è centralizzata nella classe `DataManager` (`data_manager.py`), che implementa la seguente pipeline:

```
1. load_data()         → Scarica il dataset UCI via API
2. clean_data()        → Imputa i valori nulli in stalk-root (moda)
3. feature_engineering() → Separa X e y, mappa il target in valori numerici
4. build_preprocessor()  → Costruisce il ColumnTransformer con OneHotEncoder
5. split_data()        → Train/Test split stratificato (80/20, random_state=12)
```

### Preprocessing

Tutte le feature sono categoriche, quindi il preprocessing applica **One-Hot Encoding** tramite `sklearn.compose.ColumnTransformer`:

```python
ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), X.columns)
    ]
)
```

- `drop="first"` elimina la prima categoria per evitare la multicollinearità
- `handle_unknown="ignore"` gestisce in modo sicuro eventuali categorie non viste in fase di test
- Lo split è **stratificato** per preservare la distribuzione delle classi nel train e nel test set

---

## 🤖 Modelli Implementati

### 1. XGBoost Classifier

Implementato tramite una `sklearn.Pipeline` che combina preprocessore e classificatore.

**Configurazione:**
```python
XGBClassifier(
    eval_metric="logloss",
    random_state=12
)
```

XGBoost è un algoritmo di **gradient boosting** su alberi decisionali. Costruisce iterativamente un ensemble di alberi deboli (weak learners), dove ogni nuovo albero corregge gli errori residui del precedente. La funzione di loss ottimizzata è la **log-loss** (binary cross-entropy), adatta alla classificazione binaria.

---

### 2. Logistic Regression

Anch'essa inserita in una `Pipeline` con il preprocessore.

**Configurazione:**
```python
LogisticRegression()  # parametri di default scikit-learn
```

La regressione logistica è un modello lineare che stima la **probabilità** di appartenenza a una classe tramite la funzione sigmoide:

```
P(y=1 | x) = 1 / (1 + e^(-wᵀx))
```

La classificazione avviene con soglia a 0.5. Nonostante la semplicità, è un baseline molto solido e interpretabile.

---

### 3. Rete Neurale (Deep Learning)

Implementata in `DeepLearningModel` (`deep_learning.py`) utilizzando **TensorFlow/Keras**.

**Architettura:**

```
Input   → [n_features OHE]
Dense   → 128 nodi, ReLU
Dropout → 0.2
Dense   → 64 nodi, ReLU
Dropout → 0.2
Dense   → 32 nodi, ReLU
Output  → 1 nodo, Sigmoid
```

**Compilazione:**
```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "Precision", "Recall"]
)
```

**Training:**
```python
EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
# max epochs=300, batch_size=32, validation_split=0.2
```

Il modello utilizza **Early Stopping** per interrompere il training quando la `val_loss` smette di migliorare per 20 epoche consecutive, ripristinando automaticamente i pesi migliori.

> **Nota:** A differenza dei modelli ML, la rete neurale applica il `fit_transform` del preprocessore internamente alla classe, permettendo una gestione separata rispetto alla Pipeline di scikit-learn.

---

## 📈 Risultati

Per ogni modello viene generata e salvata una **confusion matrix** nella cartella `risultati/`.

Le confusion matrix mostrano la distribuzione di:
- **True Positives (TP):** funghi commestibili classificati correttamente
- **True Negatives (TN):** funghi velenosi classificati correttamente
- **False Positives (FP):** funghi velenosi classificati come commestibili ⚠️
- **False Negatives (FN):** funghi commestibili classificati come velenosi

Le metriche riportate a console includono per ciascun modello:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Classification Report** completo per classe

| Confusion Matrix | |
|---|---|
| ![XGBoost](risultati/XGBClassifier_confusion_matrix.png) | ![Logistic Regression](risultati/Logistic%20Regressor_confusion_matrix.png) |
| ![Deep Learning](risultati/Deep%20Learning_confusion_matrix.png) | |

---

## 💾 Salvataggio di Modelli e Risultati

Al termine dell'esecuzione, tutti gli artefatti vengono serializzati nella cartella `modelli/` tramite `joblib`:

| File | Contenuto |
|---|---|
| `modelli/preprocessor.pkl` | Il `ColumnTransformer` fittato sul training set |
| `modelli/xgb_model.pkl` | Pipeline completa XGBoost (preprocessor + classifier) |
| `modelli/logistic_regression.pkl` | Pipeline completa Logistic Regression |
| `modelli/deep_learning_model.pkl` | Modello Keras serializzato con joblib |

Le **confusion matrix** (`.png`) vengono salvate automaticamente in `risultati/` durante la fase di valutazione.

### Caricare un modello salvato

```python
import joblib

# Caricare il preprocessore
preprocessor = joblib.load("modelli/preprocessor.pkl")

# Caricare e usare XGBoost
xgb_model = joblib.load("modelli/xgb_model.pkl")
y_pred = xgb_model.predict(X_new)

# Caricare la rete neurale
dl_model = joblib.load("modelli/deep_learning_model.pkl")
y_pred_prob = dl_model.predict(preprocessor.transform(X_new))
y_pred = (y_pred_prob > 0.5).astype(int)
```

---

## 🛠️ Tecnologie Utilizzate

| Libreria | Utilizzo |
|---|---|
| `ucimlrepo` | Download automatico del dataset UCI |
| `pandas` | Manipolazione e gestione dei dati |
| `scikit-learn` | Preprocessing, ML models, metriche |
| `xgboost` | Gradient Boosting Classifier |
| `tensorflow` / `keras` | Rete neurale deep learning |
| `matplotlib` / `seaborn` | Visualizzazione confusion matrix |
| `joblib` | Serializzazione dei modelli |
