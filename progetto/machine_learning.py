import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



class MachineLearningModels:
    def __init__(self, preprocessor, X_train, X_test, y_train, y_test):
        self.preprocessor = preprocessor

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.xgb_model = None
        self.logistic_regression_model = None

    #valutazione delle metriche del modello
    def evaluate_model(self, model_name, y_pred):
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f"\n{model_name}:")
        print("Accuracy:", acc)
        print("\nClassification Report:\n")
        print(report)

    #stampa delle features utilizzate dal modello
    def print_feature_names(self, model):
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        print("\nFeature usate dal modello:")
        print(feature_names)

    #training della linear regression
    def train_xgboost_classifier(self):
        model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=12
                ))
            ]
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, "XGBClassifier")



        self.evaluate_model("XGBoost Classifier", y_pred)
        self.xgb_model = model
        #self.print_feature_names(model)

    def train_logistic_regression(self):
        model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", LogisticRegression(
                ))
            ]
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, "Logistic Regressor")

        self.evaluate_model("Logistic Regression", y_pred)
        self.logistic_regression_model = model

    def plot_confusion_matrix(self, cm, nome):
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.title(nome)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()
        f.savefig(f"risultati/{nome}_confusion_matrix.png")

    def save_models(self):
        joblib.dump(self.xgb_model, "modelli/xgb_model.pkl")
        joblib.dump(self.logistic_regression_model, "modelli/logistic_regression.pkl")



