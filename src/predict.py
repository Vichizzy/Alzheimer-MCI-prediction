"""
MCI Conversion Predictor — core prediction library.

Loads the trained dual-model bundle and provides a clean API for making
24-month dementia conversion risk estimates for individual MCI patients.

Usage
-----
    from src.predict import load_predictor, predict_patient

    predictor = load_predictor("models/mci_conversion_predictor.joblib")
    result = predict_patient(patient_dict, predictor)
    print(result['probability'], result['risk_band'], result['model_used'])
"""

import os
import joblib
import numpy as np
import pandas as pd


def load_predictor(bundle_path: str) -> dict:
    """
    Load the trained dual-model bundle from disk.

    Parameters
    ----------
    bundle_path : str
        Path to mci_conversion_predictor.joblib

    Returns
    -------
    dict
        The predictor bundle containing models, feature lists, routing rule,
        recalibrated threshold, and performance metadata.
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Model bundle not found at '{bundle_path}'.\n"
            "See models/README.md for instructions on obtaining the models."
        )
    return joblib.load(bundle_path)


def predict_patient(
    patient_dict: dict,
    predictor: dict,
    use_recalibrated_threshold: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Predict 24-month dementia conversion risk for a single MCI patient.

    The system automatically selects the 26-feature or 16-feature model
    based on how many 'premium' features (RAVLT, ADAS, MRI volumes) the
    caller provides. Any missing features are imputed from training medians.

    Parameters
    ----------
    patient_dict : dict
        {feature_name: value} for any subset of the 26 model features.
        Missing features are accepted — the pipeline imputes them.
        Pass np.nan or omit the key to signal a missing measurement.
    predictor : dict
        Bundle returned by load_predictor().
    use_recalibrated_threshold : bool
        If True, uses the NACC-recalibrated threshold (0.261) instead of
        0.5 for binary classification. Recommended when applying the model
        to diverse populations outside ADNI.
    verbose : bool
        If True, prints a formatted risk report to stdout.

    Returns
    -------
    dict with keys:
        probability      : float  — P(convert within 24 months), 0–1
        risk_band        : str    — 'HIGH', 'MODERATE', or 'LOWER'
        model_used       : str    — '26-feature' or '16-feature'
        threshold_used   : float  — decision threshold for binary class
        predicted_label  : int    — 1 = converter, 0 = stable
        features_provided: list   — feature names the caller supplied
        features_imputed : list   — feature names filled from training medians
        model_auc        : float  — cross-validated AUC of the selected model
    """
    full_features   = predictor["full_features"]
    reduced_features = predictor["reduced_features"]
    full_only       = predictor["full_only_features"]

    # Routing: count how many premium features the caller provided
    premium_provided = [
        f for f in full_only
        if f in patient_dict and patient_dict[f] is not None
        and not (isinstance(patient_dict[f], float) and np.isnan(patient_dict[f]))
    ]
    use_full = len(premium_provided) >= (len(full_only) / 2)

    if use_full:
        model    = predictor["full_model"]
        features = full_features
        model_name = "26-feature (full workup)"
        model_auc  = predictor["performance"]["full_model_adni_cv_auc"]
    else:
        model    = predictor["reduced_model"]
        features = reduced_features
        model_name = "16-feature (accessible)"
        model_auc  = predictor["performance"]["reduced_model_adni_cv_auc"]

    # Build one-row DataFrame in the exact feature order the model expects
    row = {f: patient_dict.get(f, np.nan) for f in features}
    X = pd.DataFrame([row])[features]

    # Predict
    probability = float(model.predict_proba(X)[0, 1])

    threshold = (
        predictor["nacc_recalibrated_threshold"]
        if use_recalibrated_threshold
        else predictor["default_threshold"]
    )
    predicted_label = int(probability >= threshold)

    # Risk bands
    if probability >= 0.66:
        risk_band = "HIGH"
        risk_msg  = "High conversion risk — consider close monitoring and early intervention planning."
    elif probability >= 0.33:
        risk_band = "MODERATE"
        risk_msg  = "Moderate conversion risk — warrants attentive follow-up."
    else:
        risk_band = "LOWER"
        risk_msg  = "Lower conversion risk within the 24-month window — routine monitoring."

    features_provided = [f for f in features if f in patient_dict
                         and patient_dict[f] is not None
                         and not (isinstance(patient_dict.get(f), float) and np.isnan(patient_dict[f]))]
    features_imputed  = [f for f in features if f not in features_provided]

    if verbose:
        print("=" * 58)
        print("  MCI CONVERSION RISK ESTIMATE")
        print("=" * 58)
        print(f"  Model : {model_name}  (CV AUC {model_auc})")
        print(f"  Reason: {len(premium_provided)}/{len(full_only)} premium features provided")
        print()
        print(f"  Conversion probability (24 months): {probability*100:.1f}%")
        print(f"  Stable probability:                 {(1-probability)*100:.1f}%")
        print(f"  Threshold used: {threshold:.3f}  →  Predicted: {'Converter' if predicted_label else 'Stable'}")
        print()
        print(f"  Risk band: {risk_band}")
        print(f"  {risk_msg}")
        if features_imputed:
            print()
            print(f"  NOTE: {len(features_imputed)} feature(s) not provided were imputed")
            print(f"        from training medians: {features_imputed}")
        print()

    return {
        "probability":       probability,
        "risk_band":         risk_band,
        "model_used":        model_name,
        "threshold_used":    threshold,
        "predicted_label":   predicted_label,
        "features_provided": features_provided,
        "features_imputed":  features_imputed,
        "model_auc":         model_auc,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict MCI-to-dementia conversion risk for a single patient."
    )
    parser.add_argument(
        "--bundle", default="models/mci_conversion_predictor.joblib",
        help="Path to mci_conversion_predictor.joblib"
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Use NACC-recalibrated threshold (0.261) for diverse populations")
    args = parser.parse_args()

    predictor = load_predictor(args.bundle)

    # Example patient — edit values for your use case
    example = {
        "MMSCORE":   26,
        "CDRSB":      2.5,
        "FAQTOTAL":   8,
        "GDTOTAL":    3,
        "CATANIMSC":  14,
        "LIMMTOTAL":  8,
        "LDELTOTAL":  4,
        "TRAASCOR":   45,
        "TRABSCOR":   140,
        "PTEDUCAT":   16,
        "AGE":        74,
        "APOE4_count": 1,
        "sex_female": 1,
        "race_nonwhite": 0,
        "ethnicity_hispanic": 0,
        "married":    1,
    }

    predict_patient(example, predictor,
                    use_recalibrated_threshold=args.recalibrate,
                    verbose=True)
