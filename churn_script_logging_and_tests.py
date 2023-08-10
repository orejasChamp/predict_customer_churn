"""
Unit test for functions from churn_library.py.

Author: Rachel Guha
Date: 08 August 2023
"""

# Import packages.
import os
import logging
import pandas as pd
from churn_library import Model

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Set up basic logging configuration.
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    Test data import, checking for a non-zero number or rows and columns.
    '''
    try:
        churn_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: ERROR - The file wasn't found")
        raise err

    try:
        assert churn_df.shape[0] > 0
        assert churn_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: ERROR - The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    Test perform eda function by checking for saved image files.
    '''
    try:
        perform_eda()
        logging.info("Testing perform_eda: SUCCESS")

    except Exception as err:
        logging.error("Testing perform_eda: ERROR - type %s", type(err))
        raise err

    try:
        assert os.path.isfile('images/eda/Churn.png')
        assert os.path.isfile('images/eda/Customer_Age.png')
        assert os.path.isfile('images/eda/Heatmap.png')
        assert os.path.isfile('images/eda/Marital_Status.png')
    except AssertionError as err:
        logging.error(
            "Testing perform_eda - image generation: ERROR - Image files missing.")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper by checking for the number of encoded columns.
    '''
    try:
        churn_df = encoder_helper("Churn")
        logging.info("Testing encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing encoder_helper: ERROR - Check for missing categorical features")
        raise err

    try:
        assert sum(churn_df.columns.str.contains('_Churn')) == 5
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: ERROR - Number of encoded columns is incorrect.")
        raise err


def test_feature_engineering(feature_engineering):
    '''
    Test feature_engineering by checking for the existance of
    x_train, x_test, y_train, y_test dataframes.
    '''
    try:
        X_train, X_test, y_train, y_test = feature_engineering("Churn")
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - Data split was not performed.")
        raise err


def test_train_models(train_models):
    '''
    test train_models by checking for pickled models.
    '''
    try:
        train_models()
        assert os.path.isfile('models/logistic_model.pkl')
        assert os.path.isfile('models/rfc_model.pkl')
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Training not successful.  Models not saved.")
        raise err


if __name__ == "__main__":
    MODEL = Model()
    test_import(MODEL.import_data)
    test_eda(MODEL.perform_eda)
    test_encoder_helper(MODEL.encoder_helper)
    test_feature_engineering(MODEL.feature_engineering)
    test_train_models(MODEL.train_models)
