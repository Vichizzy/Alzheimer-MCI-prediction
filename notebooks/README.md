# Notebooks

These notebooks contain the complete analysis pipeline, from raw data to validated models.

## Contents

### `01_ADNI_model_training.ipynb`
Full training pipeline on the ADNI cohort:
- Exploratory data analysis of ADNI clinical and imaging datasets
- Cohort construction: filtering MCI-at-baseline, building 24-month conversion labels
- Feature engineering: cognitive tests, neuropsychological battery, MRI volumes, genetics, demographics
- Feature selection and missingness analysis
- Model training: Logistic Regression, Random Forest, XGBoost
- CSF biomarker comparison (with vs. without amyloid-β, tau, phospho-tau)
- Cross-validation and bootstrap confidence intervals
- SHAP feature importance analysis
- Saving trained models to `models/`

### `02_NACC_external_validation.ipynb`
Full external validation pipeline on the NACC cohort:
- Loading ADNI-trained models from `models/`
- NACC dataset preparation and cohort construction (6,321 MCI patients)
- Feature harmonization: mapping NACC column names to ADNI equivalents
- External validation: applying ADNI-trained model to NACC without retraining
- Bootstrap confidence intervals for NACC AUC
- Threshold recalibration: calibration split → Youden's J → reporting split
- Sensitivity analysis: complete cases vs. fully imputed cohort
- Fairness/subgroup analysis: by race, sex, ethnicity, age
- Combined ADNI+NACC training experiment
- Dual-model patient prediction tool
- Packaging final bundle to `models/mci_conversion_predictor.joblib`

## Requirements

Notebooks require the raw ADNI and NACC data files, which are not included in this
repository (restricted access — see main README). To reproduce the analysis:

1. Request ADNI data access at [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. Request NACC data access at [naccdata.org](https://naccdata.org)
3. Place downloaded files in the appropriate paths (see notebook cell comments)
4. Install requirements: `pip install -r ../requirements.txt`
5. Add Jupyter: `pip install jupyter`

The trained models in `models/` can be loaded and used without running the notebooks.
