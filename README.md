# Explainable EMI Loan Decisions — Responsible & Safe AI

**External Practical Examination · Set B · AIML4001 · 25 November 2026**

**Student:** Yash Bharvada · **Roll number:** 23CS006

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Open%20Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/bharvadayash/practical)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Focus](https://img.shields.io/badge/Focus-Responsible%20AI-2E7D32)

> **Question:** When an automated EMI-loan model rejects an applicant, can we explain which model features drove the decision and check whether sensitive attributes appear among the reasons?

This project addresses **“Explaining Instant EMI-Finance Decisions at an Electronics Showroom”** using a Random Forest model trained on the UCI Statlog German Credit dataset. **SHAP** provides global and applicant-specific explanations; **LIME** provides an independent local explanation for comparison. The work also checks whether protected or sensitive attributes are highlighted in rejection reasoning.

## ▶ Run the actual notebook

### [Open the practical on Kaggle](https://www.kaggle.com/code/bharvadayash/practical)

**The executable notebook is hosted on Kaggle, not in this GitHub repository.** The former `RSA_SetB_EMI_Explainability_Kaggle.ipynb` was intentionally removed from GitHub. The repository retains the supporting Python source in [`src/emi_explainability.py`](src/emi_explainability.py).

For reproducibility, open the Kaggle notebook, attach the **German Credit** data if required, and run all cells. Kaggle's attached dataset and package environment determine whether additional input or dependency setup is needed.

## Practical objectives

| Task | Requirement | Method |
|:---:|---|---|
| **1** | Load German Credit, encode categorical data, and split 80:20 | `OrdinalEncoder`, stratified split, seed **42** |
| **2** | Train and evaluate credit-risk classifier | Random Forest: **200 trees**, `max_depth=8`; accuracy and ROC-AUC |
| **3** | Explain overall feature influence | SHAP TreeExplainer; mean absolute SHAP; summary plot and top 5 |
| **4** | Explain a predicted-bad applicant | SHAP waterfall; top 3 feature contributions |
| **5** | Cross-check the same applicant | LIME with `num_features=5`; top-3 agreement |
| **6** | Audit explanations and communicate decision | Check `age`, `personal_status`, `foreign_worker`; plain-language message |

**Target convention:** `bad` credit is the positive class (`1`).

## Observed results — Set B run

These numbers are **observed results from the supplied practical outputs**, not the example numbers printed on the exam question sheet.

| Measurement | Result |
|---|---:|
| Test accuracy | **77.00%** |
| Test ROC-AUC | **0.8113** |
| Selected rejected test applicant | Test position **2**, original row **289** |
| Model's probability of bad credit | **0.5328** |
| SHAP–LIME top-three feature agreement | **1 / 3** |

### Top five global SHAP features

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | `checking_status` | 0.0867 |
| 2 | `duration` | 0.0413 |
| 3 | `savings_status` | 0.0391 |
| 4 | `credit_amount` | 0.0258 |
| 5 | `credit_history` | 0.0258 |

### Why was this applicant rejected?

The three largest *absolute* SHAP contributions were:

| Feature | Applicant value | SHAP effect on `P(bad)` |
|---|---|---:|
| `checking_status` | `A11` | **+0.1134** |
| `duration` | `24` | **+0.0486** |
| `property_magnitude` | `A121` | **−0.0384** |

The **positive** contributions increased predicted bad-credit risk relative to the explainer baseline. The **negative** property contribution reduced predicted risk; it should **not** be described as a reason increasing rejection risk.

LIME's three leading feature names were `checking_status`, `savings_status`, and `employment`. Thus only `checking_status` appears in both leading-three sets.

**Sensitive-attribute audit:** None of the three leading SHAP feature names belongs to the specified sensitive set. **Age appeared in the LIME top five**, however, so that explanation is flagged for human/compliance review. The absence of a sensitive attribute from a short explanation does not establish model fairness.

## Project files

```text
RSA-SetB-EMI-Explainability/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── emi_explainability.py
```

The **final exam report**, `23CS006_External_Practical_Exam.pdf`, and the submission ZIP of plots/results are **separate submission artifacts**, not currently stored in this repository. The final ZIP uses the new report rather than the earlier draft.

## Reproducing and submitting

1. Open the **[Kaggle practical](https://www.kaggle.com/code/bharvadayash/practical)**.
2. Confirm the German Credit dataset input is available, and run the notebook.
3. Review the resulting model metrics, SHAP and LIME plots, explanation comparisons, and sensitivity audit.
4. Include the **aim, algorithm, program, output, conclusion, and answers to technical questions Q1–Q4** in the final PDF.
5. Submit the PDF **`23CS006_External_Practical_Exam.pdf`** and the results ZIP **separately** to Google Classroom, as required by the exam instructions.

**Source repository:** https://github.com/Yash-Bharvada/RSA-SetB-EMI-Explainability  
**Kaggle notebook:** https://www.kaggle.com/code/bharvadayash/practical

## Dependencies

Python 3, NumPy, pandas, scikit-learn, Matplotlib, SHAP, LIME, IPython. See [`requirements.txt`](requirements.txt) for minimum package versions.

```bash
python -m pip install -r requirements.txt
```

> **Responsible-AI limitation:** This is a classroom demonstration, not an approved lending system. SHAP and LIME explain the model's behavior, not the causal reasons for an applicant's creditworthiness. Explanations alone do not prove non-discrimination; fairness evaluation and human review remain important.
