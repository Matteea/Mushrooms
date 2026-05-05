
from progetto.data_manager import DataManager
from progetto.deep_learning import DeepLearningModel
from progetto.machine_learning import MachineLearningModels

import pandas as pd


def main():
    #cambio impostazioni di visualizzazione di pandas
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    #inizializzazione della classe per la gestione dei dati
    data_manager = DataManager()
    data_manager.prepare_data()
    print(data_manager.df.info())


    #inizializzazione della classe dei modelli di machine learning
    ml_models = MachineLearningModels(
        preprocessor=data_manager.preprocessor,
        X_train=data_manager.X_train,
        X_test=data_manager.X_test,
        y_train=data_manager.y_train,
        y_test=data_manager.y_test
    )

    #allenamento e risultati dei modelli
    ml_models.train_xgboost_classifier()
    ml_models.train_logistic_regression()

    #inizializzazione della classe per il deep learning
    dl_model = DeepLearningModel(
        preprocessor=data_manager.preprocessor,
        X_train=data_manager.X_train,
        X_test=data_manager.X_test,
        y_train=data_manager.y_train,
        y_test=data_manager.y_test
    )

    #allenamento e risultati del modello di deep learning
    dl_model.run()

    data_manager.save_preprocessor()
    ml_models.save_models()
    dl_model.save_models()

if __name__ == "__main__":
    main()