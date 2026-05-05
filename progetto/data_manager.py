import joblib
from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class DataManager:
    def __init__(self, test_size=0.2, random_state=12):
        self.test_size = test_size
        self.random_state = random_state

        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.preprocessor = None

    #caricamento del dataset
    def load_data(self):
        mushrooms = fetch_ucirepo(id=73)

        self.df = pd.DataFrame(mushrooms.data.features)
        edible = mushrooms.data.targets.squeeze()
        #print(edible)

        self.df["Edible"] = edible

        print("Shape iniziale:", self.df.shape)


    #pulizia del dataset
    def clean_data(self):
        #print(self.df.isnull().sum())
        mode = self.df['stalk-root'].mode()[0]
        self.df['stalk-root'] = self.df['stalk-root'].fillna(mode)
        #print(self.df.isnull().sum())
        print("Shape dopo pulizia:", self.df.shape)

    #ingegnerizzazione delle feature
    def feature_engineering(self):
        df = self.df.copy()

        self.X = df.drop(columns=["Edible"])
        self.y = df["Edible"].map({"e": 1, "p": 0})


    #preparazione del preprocessor
    def build_preprocessor(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                 self.X.columns)
            ],
            remainder="drop"
        )

    #splitting dei dati
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state
        )

    #preparazione dei dati
    def prepare_data(self):
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.build_preprocessor()
        self.split_data()

    def save_preprocessor(self):
        joblib.dump(self.preprocessor, "modelli/preprocessor.pkl")