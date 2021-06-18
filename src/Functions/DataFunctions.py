from sklearn import preprocessing
import joblib

from . import defaults

def labelEncode(x):
    """
    Description: This function computes the label encoded values of a given array or features 
    based on the label encoders used during training.
    Return: Label encoded values of a given array or features
    """
    for feature in defaults.categorical:
        le = joblib.load("model_training/joblib/le." + feature + ".joblib")
        x[feature] = le.fit_transform(x[feature])
    return x