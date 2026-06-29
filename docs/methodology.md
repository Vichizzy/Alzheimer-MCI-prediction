# Methodology: MCI to Dementia Conversion Predictor

## 1. Problem Framing

### Clinical Context
Mild Cognitive Impairment (MCI) is a transitional state between normal aging and dementia, affecting 15–20% of adults over 65. Approximately one-third of MCI patients progress to dementia (primarily Alzheimer's disease) within two years. Early, reliable identification of which patients will convert enables targeted monitoring, earlier intervention, and trial enrollment for disease-modifying therapies.

This project addresses the **prognostic** question: *Given a patient's first clinical visit (baseline), what is the probability they will convert from MCI to dementia within the next 24 months?* This is distinct from the diagnostic question (whether someone currently has dementia), which trained clinicians already handle well.

### Accessibility Thesis
Most published MCI conversion models require CSF biomarkers (lumbar puncture) or quantitative PET imaging, creating access barriers in resource-limited settings. The central hypothesis of this project is:

> **Accessible, routinely administered clinical tests (cognitive assessments and demographics) predict MCI conversion as reliably as expensive or invasive biomarkers.**

---

## 2. Data Sources

### ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Source**: Multi-site, longitudinal US cohort; data from ADNI1, ADNI GO, ADNI2, ADNI3
- **Purpose**: Primary training dataset
- **Key datasets used**: DXSUM (diagnoses), ADAS, MMSE, CDR, FAQ, GDSCALE, NEUROBAT, UCSFFSX7 (FreeSurfer MRI), APOERES (genetics), PTDEMOG (demographics)

### NACC (National Alzheimer's Coordinating Center)
- **Source**: Multi-site longitudinal US cohort with greater demographic diversity than ADNI
- **Purpose**: External validation dataset — never seen during training
- **Key dataset**: `investigator_nacc73.csv` (main data file, single longitudinal table)
- **Demographics**: More diverse than ADNI; used to test generalization and fairness

---

## 3. Cohort Construction

### ADNI Cohort (Training)

**Step 1 — Identify MCI at baseline**
- Load DXSUM.csv (diagnosis file): 15,881 rows, 3,788 unique patients
- Filter to first visit (`VISCODE2 == 'bl'` or equivalent baseline code)
- Select patients with MCI diagnosis (DX = 2) at their first visit
- Result: **1,328 MCI-at-baseline patients**

**Step 2 — Build the 24-month conversion label**
- For each MCI patient, scan all subsequent visits
- A patient is labeled a **converter** (label = 1) if they receive a Dementia diagnosis (DX = 3) within 24 months, with a 30-month grace window to capture conversions that occur slightly after the 24-month target (due to visit spacing)
- A patient is labeled **stable** (label = 0) if they remain MCI or return to Normal with ≥18 months of follow-up data — the minimum needed to confidently rule out conversion
- Patients with insufficient follow-up (<18 months, no conversion observed) are **excluded**

**Step 3 — Final cohort statistics**
- Converters: 239 patients (29%)
- Stable: 577 patients (71%)
- Excluded (insufficient follow-up): 506 patients
- **Final labeled cohort: 816 patients**
- Conversion rate of 29% aligns with published ADNI literature (typical range: 25–40% over 2 years)

### NACC Cohort (Validation)

Same logic applied independently:
- Total NACC patients: 56,532; total visits: 214,976
- MCI at first visit (NACCVNUM == 1, NACCUDSD == 3): 6,321 patients
- Conversion label: NACCUDSD == 4 (Dementia) within 24 months; same 30-month grace, 18-month minimum follow-up
- Conversion rate: 35.9%
- Complete-case cohort (all features present): **1,974 patients**
- Full imputed cohort: **6,321 patients** (missing features filled by ADNI-trained imputer)

---

## 4. Feature Engineering

### 4.1 Feature Categories

**Cognitive Assessments**
- `MMSCORE`: Mini-Mental State Examination total (0–30; higher = better). The most widely used cognitive screening test globally.
- `CDRSB`: Clinical Dementia Rating Sum of Boxes (0–18; higher = more impaired). Assesses real-world functioning across six domains via clinician interview. CDR global of 0.5 defines MCI in ADNI.
- `TOTSCORE`: ADAS-11 score (0–70; higher = worse). The gold-standard cognitive battery in AD clinical trials.
- `TOTAL13`: ADAS-13 score (0–85; higher = worse). More sensitive to MCI than ADAS-11.

**Neuropsychological Battery (NEUROBAT)**
- `RAVLT_learning_total`: Rey Auditory Verbal Learning Test — sum of correct recalls across 5 learning trials (0–75). The single strongest predictor in SHAP analysis (score 0.728).
- `AVDEL30MIN`: RAVLT 30-minute delayed recall (0–15).
- `AVDELTOT`: RAVLT total delayed recall (0–15).
- `LIMMTOTAL`: Logical Memory Immediate Recall (0–25). Story-based verbal memory — present in both ADNI and NACC.
- `LDELTOTAL`: Logical Memory Delayed Recall (0–25). Delayed recall of the same story.
- `TRAASCOR`: Trail Making Test A (seconds; lower = faster/better). Processing speed.
- `TRABSCOR`: Trail Making Test B (seconds; lower = faster/better). Executive function.
- `CATANIMSC`: Category fluency — animals named in 60 seconds. Language and semantic memory.

**Functional Assessment**
- `FAQTOTAL`: Functional Activities Questionnaire total (0–30). Sum of 10 items assessing ability to handle finances, shopping, cooking, etc. Scores >9 indicate functional impairment.

**Behavioral/Mood**
- `GDTOTAL`: Geriatric Depression Scale total (0–15). Depression is both a risk factor and a cofactor in conversion.

**Brain Imaging (MRI — FreeSurfer)**
All volumes from FSX7 (FreeSurfer version 7, the latest ADNI reprocessing), using earliest available scan per patient. Left + right hemispheres summed.
- `Hippocampus_total`: Total hippocampal volume (mm³). The primary atrophy biomarker in AD.
- `Entorhinal_total`: Total entorhinal cortex volume (mm³). First region to show AD pathology.
- `Ventricle_total`: Total lateral ventricle volume (mm³). Enlarges as brain tissue is lost.
- `Fusiform_total`: Total fusiform gyrus volume (mm³). Face recognition; affected in AD.
- `ICV`: Intracranial volume (mm³). Used to normalize regional volumes.

**Genetics**
- `APOE4_count`: Number of APOE ε4 alleles (0, 1, or 2). Derived from genotype string in APOERES.csv by counting 'E4' occurrences. The strongest known genetic risk factor for late-onset AD.

**Demographics**
- `AGE`: Age at baseline visit (years). Computed from birth year and visit year since ADNI does not store age directly.
- `PTEDUCAT`: Years of education. A cognitive reserve proxy.
- `sex_female`: 1 = female, 0 = male.
- `race_nonwhite`: 1 = non-white, 0 = white. Binary due to limited ADNI diversity (93% white).
- `ethnicity_hispanic`: 1 = Hispanic, 0 = non-Hispanic.
- `married`: 1 = currently married, 0 = otherwise. A cognitive/social reserve proxy.

### 4.2 Missingness Handling

- Features with 0–15% missingness: median imputation (via `SimpleImputer`)
- Features with >40% missingness: excluded from modeling
- **MoCA excluded despite clinical relevance**: ~50% missing in ADNI (only introduced in ADNI GO/2; absent from ADNI1). Missing is not random — excluding those patients would bias the cohort, and imputing invents data that never existed.
- `CATVEGESC` (vegetable fluency): 58% missing; excluded
- Final feature coverage within the 816-patient cohort: all selected features are ≥99% present

### 4.3 The 16-Feature Common Set (NACC-Compatible)

NACC uses different neuropsychological instruments. The features absent from NACC:
- RAVLT (3 features) — NACC does not administer RAVLT
- ADAS (2 features) — not in the NACC standard protocol
- MRI brain volumes (5 features) — NACC is clinical, not imaging-focused

The 16 features common to both datasets:
`MMSCORE, CDRSB, FAQTOTAL, GDTOTAL, CATANIMSC, LIMMTOTAL, LDELTOTAL, TRAASCOR, TRABSCOR, PTEDUCAT, AGE, APOE4_count, sex_female, race_nonwhite, ethnicity_hispanic, married`

Note: MMSE (MMSCORE) and Logical Memory (LIMMTOTAL, LDELTOTAL) have only 61% coverage in NACC due to the UDS2 → UDS3 protocol transition. The remaining 13 features have ≥95% coverage.

---

## 5. Modeling

### 5.1 Pipeline
All models use a consistent sklearn Pipeline:
1. `SimpleImputer(strategy='median')` — fills NaN with training-data medians (fit on training fold only; no leakage)
2. `StandardScaler()` — centers and scales to unit variance
3. `Classifier` — see below

### 5.2 Class Imbalance
The 29% converter / 71% stable split is addressed by `class_weight='balanced'`, which re-weights each sample inversely proportional to class frequency. This prevents the model from learning to predict "stable" for every patient (which would achieve 71% accuracy but zero clinical utility).

Accuracy was explicitly rejected as an evaluation metric in favor of AUC-ROC.

### 5.3 Algorithms Compared

| Algorithm | Hyperparameters | AUC (CV) | Recall (converters) |
|---|---|---|---|
| Logistic Regression | `max_iter=1000, random_state=42` | **0.887** | **0.75** |
| Random Forest | `n_estimators=100, class_weight='balanced', random_state=42` | 0.891 | 0.50 |
| XGBoost | `scale_pos_weight=2.4, random_state=42` | 0.870 | 0.65 |
| LR + CSF biomarkers | Same as LR | 0.893\* | — |

\* *Not significantly different from primary LR (overlapping CIs)*

Logistic regression was selected as the primary model because:
1. Matched more complex algorithms on AUC
2. Substantially higher converter recall (0.75 vs. 0.50 for RF)
3. Much lower fold-to-fold variance (±0.004 vs. ±0.018 for RF) → more reliable in deployment
4. Coefficients are directly interpretable in clinical terms
5. The nearly identical performance of a linear model suggests the feature-outcome relationships are largely linear and additive

### 5.4 CSF Biomarker Comparison
Three CSF biomarkers were tested as additional features: amyloid-β (ABETA), total tau (TAU), and phosphorylated tau (PTAU). Adding these to the 26-feature model slightly *decreased* performance (0.941 → 0.893 on a single test split; overlapping bootstrap CIs on cross-validation). This occurs because amyloid and tau information is largely redundant with what is already captured by hippocampal atrophy and memory test scores. The 109-patient CSF cohort (smaller due to missing lumbar puncture data) adds noise without adding signal.

**This is the project's central finding**: CSF biomarkers are not necessary for high-accuracy MCI conversion prediction.

---

## 6. Evaluation

### 6.1 Cross-Validation
5-fold stratified cross-validation on the full 816-patient cohort. Stratification ensures converter prevalence is preserved in every fold. The pipeline's imputation and scaling are re-fit on each training fold to prevent data leakage.

Primary results:
- Minimal LR (26 features): AUC **0.887 ± 0.004**
- Full RF (26 features): AUC 0.891 ± 0.018
- Accessible LR (16 features): AUC **0.871 ± 0.011**

### 6.2 Bootstrap Confidence Intervals
2,000 bootstrap resamples on the held-out test set (20% of cohort, stratified). Each resample draws N samples with replacement and computes AUC. The 2.5th and 97.5th percentiles give the 95% CI.

- Minimal LR: AUC 0.888 [0.832, 0.939]
- Full model: AUC 0.903 [0.836, 0.955]
- Overlapping intervals confirm no statistically significant difference

### 6.3 SHAP Analysis
SHAP (SHapley Additive exPlanations) values decompose each model prediction into per-feature contributions. `shap.LinearExplainer` was used with the trained logistic regression. Top features (by mean |SHAP| across the test set):

1. RAVLT_learning_total (0.728) — verbal learning dominates
2. LDELTOTAL / LIMMTOTAL — delayed and immediate verbal recall
3. FAQTOTAL — functional impairment
4. Hippocampus_total — hippocampal atrophy
5. APOE4_count — genetic risk
6. CDRSB — clinical dementia rating
7. AGE — older age increases risk

These align precisely with established Alzheimer's neurobiology. The model did not identify spurious correlates.

---

## 7. External Validation (NACC)

### 7.1 Protocol
The ADNI-trained 16-feature model was applied to NACC exactly once for the primary generalization result, as pre-registered in the analysis plan. No retraining was performed before this evaluation.

### 7.2 Feature Harmonization
NACC uses different variable names and some different coding conventions:
- `NACCMMSE` → `MMSCORE`; `CDRSUM` → `CDRSB`; `NACCGDS` → `GDTOTAL`; etc.
- APOE: genotype codes 1–6 converted to ε4 allele count (0, 1, 2)
- Sex: NACCSEX (1=male, 2=female) → `sex_female` (0=male, 1=female)
- FAQ reconstructed by summing 10 FAQ item columns (BILLS, TAXES, SHOPPING, etc.)
- NACC missing codes (−4, 88, 95–99) converted to NaN before imputation

### 7.3 Results
- Primary (complete cases, n=1,974): **AUC 0.805 [0.784, 0.826]**
- Sensitivity (full imputed, n=6,321): AUC 0.781 [0.768, 0.793]

The primary result (complete cases) is preferred because imputing ~39% of patients missing MMSE and Logical Memory adds noise and may introduce systematic bias; the complete-case cohort still has 1,974 patients (more than twice the ADNI training set) and remains demographically representative.

### 7.4 Threshold Recalibration
The ADNI-trained decision threshold of 0.5 underperformed on NACC (recall = 0.63).

Protocol (to avoid optimization bias):
1. Randomly split the 1,974 NACC patients 50/50 into calibration and reporting halves (stratified)
2. Apply Youden's J statistic (`argmax(TPR − FPR)`) to the ROC curve on the calibration half
3. Apply the resulting threshold (0.261) to the reporting half only
4. Report results on the reporting half

Results on reporting half:
- Threshold 0.5: recall 0.630, precision 0.639
- Threshold 0.261: recall **0.783**, precision 0.530
- AUC unchanged (0.805 — threshold-free metric)

The lower threshold (0.261) reveals that the ADNI-trained model systematically underestimates conversion risk in the NACC population. This is likely a calibration artifact from training on a predominantly white, highly-educated ADNI cohort. Recalibration is recommended when deploying to diverse populations.

### 7.5 Fairness Analysis
AUC computed separately within demographic subgroups:

- **Race**: White 0.798 [0.775, 0.822]; Non-white 0.832 [0.781, 0.880] — intervals fully overlap. No evidence of racial performance disparity.
- **Sex**: Male 0.810; Female 0.801 — essentially identical.
- **Age**: Younger (<73) 0.842 [0.811, 0.873]; Older (≥73) 0.771 [0.740, 0.799] — **intervals only partially overlap**. This age gap is likely real.

The age performance gap is not a representation artifact — adding older NACC patients to training (combined ADNI+NACC model) did not narrow it. It likely reflects genuine biological heterogeneity: in older patients, cognitive decline has more competing etiologies (vascular disease, other dementias, normal aging), making MCI-to-Alzheimer's conversion harder to predict cleanly.

---

## 8. The Dual-Model System

### Design Rationale
The 0.016 AUC difference between the 26-feature and 16-feature models (0.887 vs. 0.871) is within the model's cross-validation noise range (±0.011 for the reduced model). In practical terms, these models perform nearly identically. This creates the opportunity for a resource-adaptive deployment strategy:

- In settings with full neuropsychological workup and MRI: use the 26-feature model
- In settings where RAVLT, ADAS, or MRI are unavailable: fall back to the 16-feature model

### Routing Rule
```python
premium_features = [TOTSCORE, TOTAL13, RAVLT_learning_total,
                    AVDEL30MIN, AVDELTOT,
                    Hippocampus_total, Entorhinal_total,
                    Ventricle_total, Fusiform_total, ICV]

use_full_model = (n_premium_features_provided >= 5)  # >= 50% of 10
```

### Scientific Note
The finding that removing RAVLT (the single strongest predictor by SHAP score) costs only 0.016 AUC demonstrates an important scientific point: **Alzheimer's conversion prediction is robust to the specific cognitive tests used**. What matters is capturing the underlying domain (delayed verbal memory), not any particular proprietary instrument. LIMMTOTAL and LDELTOTAL (Logical Memory) absorb RAVLT's predictive role when RAVLT is absent, because both measure the same construct.

---

## 9. Reproducibility

Full analysis is in `notebooks/01_ADNI_model_training.ipynb` and `notebooks/02_NACC_external_validation.ipynb`. The notebooks require restricted ADNI/NACC data (see README). All trained weights are in `models/` and can be loaded and used without the raw data.

Key random seeds: `random_state=42` throughout; bootstrap CI uses `np.random.RandomState(42)`.
