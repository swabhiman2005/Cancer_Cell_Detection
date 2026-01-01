import os
import sys
#import pickle
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    This function saves a python object (like a model or scaler) 
    into a pickle file (.pkl).
    """
    try:
        dir_path = os.path.dirname(file_path)

        # Create folder if it doesn't exist (e.g., 'artifacts' folder)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            # pickle.dump(obj, file_obj)
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    '''
    This function performs hyperparameter tuning using GridSearchCV 
    and evaluates multiple models to find the best performing one.
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Initialize GridSearchCV to find the best hyperparameters
            logging.info(f"Started Hyperparameter tuning for: {list(models.keys())[i]}")
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Re-train the model using the best found parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on the test dataset
            y_test_pred = model.predict(X_test)

            # Calculate the accuracy score for model evaluation
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Store the score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        logging.info("Model evaluation completed successfully")
        return report

    except Exception as e:
        raise CustomException(e, sys)
