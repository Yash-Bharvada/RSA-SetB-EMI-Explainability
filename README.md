# Responsible & Safe AI — EMI Loan Explainability

**AIML4001 · External Practical Examination · Set B · 25 November 2026**

An explainable machine-learning workflow for instant EMI lending decisions in an electronics showroom. The project trains a Random Forest on the German Credit dataset, explains predictions using SHAP and LIME, and audits the rejection explanations for sensitive attributes.

## Kaggle notebook

**[`RSA_SetB_EMI_Explainability_Kaggle.ipynb`](RSA_SetB_EMI_Explainability_Kaggle.ipynb)** is the full, importable practical. It covers:

1. Load and validate German Credit data, encode categories with `OrdinalEncoder`, and make an 80:20 stratified split with `random_state=42`.
2. Train `RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)`, and calculate accuracy and ROC-AUC.
3. Calculate class-`bad` global SHAP importance, render summary plots, and list the top five features.
4. Explain one actually rejected test applicant with a local SHAP waterfall plot and list the three strongest contributions.
5. Explain the same applicant with LIME (`num_features=5`) and compare the three leading reasons.
6. Generate a plain-language rejection explanation and audit whether `age`, `personal_status`, or `foreign_worker` appear.

The notebook additionally generates results for the four technical questions in the examination paper, as well as a PDF report and a submission ZIP containing the run outputs.

## Run in Kaggle

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and import `RSA_SetB_EMI_Explainability_Kaggle.ipynb`.
2. Use **Add Input** to attach the genuine **UCI Statlog German Credit** dataset. The exam sheet does *not* include the dataset. The notebook can locate several common German Credit filenames and layouts.
3. Enable **Internet** if `lime` needs installation. Run all cells.
4. In the notebook configuration cell, replace `GITHUB_REPOSITORY_URL` with this repository's actual URL **before generating the final PDF**.
5. Download the PDF and output ZIP from `/kaggle/working/` and submit them separately on Google Classroom.

## Repository layout

```text
.
├── RSA_SetB_EMI_Explainability_Kaggle.ipynb  # Complete notebook
├── README.md                                 # Project setup and description
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Ignore generated outputs and caches
└── src/
    └── emi_explainability.py                # Python export of notebook code
```

## Outputs after running

Outputs include test performance metrics, SHAP summary and waterfall plots, LIME explanations, the sensitive-attribute audit, Q1–Q4 answers, the practical PDF report and a ZIP of execution artifacts. Numerical results depend on the actual dataset supplied and must not be copied from the question paper's illustrative output.

## Requirements

Python 3 with pandas, NumPy, scikit-learn, Matplotlib, SHAP, LIME, and IPython. Kaggle supplies many packages; the notebook attempts installation of missing explanation packages.

## Responsible-use note

This is an academic explainability exercise, **not** a production lending system. Model explanations are associations with the model output, not proven causal reasons. Sensitive features must be reviewed before use in real lending decisions.