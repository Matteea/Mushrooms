import joblib
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DeepLearningModel:
    def __init__(self, preprocessor, X_train, X_test, y_train, y_test):
        self.preprocessor = preprocessor

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.X_train_processed = None
        self.X_test_processed = None
        self.model = None

    #preprocess dei dati
    def preprocess_data(self):
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)

    #costruzione della rete neurale, 3 hidden layer da 64, 32 e 16 nodi
    def build_model(self):
        self.model = Sequential([
            Input(shape=(self.X_train_processed.shape[1],)),

            Dense(128, activation="relu"),
            Dropout(0.2),

            Dense(64, activation="relu"),
            Dropout(0.2),

            Dense(32, activation="relu"),

            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall"]
        )

    #allenamento del modello fino al raggiungimento di uno specifico valore di val_loss
    def train_model(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        self.model.fit(
            self.X_train_processed,
            self.y_train,
            validation_split=0.2,
            epochs=300,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )


    #valutazione delle metriche del modello
    def evaluate_model(self):
        # probabilità → classi
        y_pred_prob = self.model.predict(self.X_test_processed).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, "Deep Learning")

        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print("\nDeep Learning (Binary Classification):")
        print("Accuracy:", acc)
        print("\nClassification Report:\n")
        print(report)

    #esecuzione completa della classe
    def run(self):
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()

    def plot_confusion_matrix(self, cm, nome):
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.title(nome)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()
        f.savefig(f"risultati/{nome}_confusion_matrix.png")

    def save_models(self):
        joblib.dump(self.model, "modelli/deep_learning_model.pkl")
