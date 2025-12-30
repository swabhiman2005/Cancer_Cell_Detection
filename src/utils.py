import os
import sys
#import pickle
import dill
import numpy as np
import pandas as pd
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

#def evaluate_models(X_train, y_train, X_test, y_test, models):
#    """
#    This is an extra helpful function to train and test 
 #   multiple models at once and return their scores.
 #   """
 #   try:
#        report = {}
#
#        for i in range(len(list(models))):
#            model = list(models.values())[i]
            
            # Train the model
#            model.fit(X_train, y_train)

            # Make predictions
#            y_test_pred = model.predict(X_test)

            # Calculate accuracy score
#            test_model_score = accuracy_score(y_test, y_test_pred)

#            report[list(models.keys())[i]] = test_model_score

#        return report

#    except Exception as e:
#        raise CustomException(e, sys)