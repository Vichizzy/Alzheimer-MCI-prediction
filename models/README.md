# Models

This directory contains the trained model files for the MCI Conversion Predictor.

## Contents

| File | Description | Size |
|---|---|---|
| `mci_conversion_predictor.joblib` | **Primary bundle** — both models + routing logic + metadata | ~3.2 MB |
| `minimal_model_LR.joblib` | 26-feature Logistic Regression (primary model, AUC 0.887) | ~3 KB |
| `full_model_RF.joblib` | 26-feature Random Forest (comparison model, AUC 0.891) | ~3.2 MB |
| `reduced_common_model_LR.joblib` | 16-feature Logistic Regression (accessible model, AUC 0.871) | ~3 KB |
| `full_features.joblib` | Ordered list of the 26 feature names | <1 KB |
| `minimal_features.joblib` | Same — named `minimal` in original training code | <1 KB |

## The Primary Bundle

The Streamlit app and `src/predict.py` both load `mci_conversion_predictor.joblib`.
This single file contains:

```python
bundle = {
    'full_model':     <26-feature LogisticRegression Pipeline>,
    'reduced_model':  <16-feature LogisticRegression Pipeline>,
    'full_features':  [...],   # ordered list of 26 feature names
    'reduced_features': [...], # ordered list of 16 feature names
    'full_only_features': [...], # 10 features only in the full model
    'routing_rule':   'Use full_model if >=50% of full_only_features present',
    'default_threshold': 0.5,
    'nacc_recalibrated_threshold': 0.261,  # for diverse populations
    'performance': {
        'full_model_adni_cv_auc':    0.887,
        'reduced_model_adni_cv_auc': 0.871,
        'reduced_model_nacc_auc':    0.805,
        'reduced_model_nacc_auc_ci': (0.784, 0.826),
        'reduced_model_nacc_recall_recalibrated': 0.783,
    },
    'task':    'Predict conversion from MCI to dementia within 24 months',
    'output':  'Probability of conversion (0-1)',
    'note':    'Prognostic tool for already-diagnosed MCI patients; not diagnostic',
    'created': '2026-06-25',
}
```

## Each Pipeline's Structure

Both the full and reduced models are identical sklearn Pipelines:

```
SimpleImputer(strategy='median')   → fills NaN with training medians
StandardScaler()                   → centers and scales features
LogisticRegression(
    class_weight='balanced',       → corrects 29/71% class imbalance
    max_iter=1000,
    random_state=42
)
```

## Important Note on Data

These model files contain **only trained weights derived from ADNI and NACC**.
They do not contain any patient-level data. Raw ADNI/NACC data is private
and not included in this repository. See the main README for data access links.
