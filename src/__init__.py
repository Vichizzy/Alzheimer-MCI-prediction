"""
MCI Conversion Predictor — prediction library.
Load the trained model bundle and predict dementia conversion risk for individual patients.
"""
from .predict import load_predictor, predict_patient

__all__ = ["load_predictor", "predict_patient"]
