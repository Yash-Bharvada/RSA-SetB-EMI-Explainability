# -*- coding: utf-8 -*-
"""Python export of the RSA Set B Kaggle notebook. Run with Kaggle inputs configured."""



# %% [cell 2]
from pathlib import Path
import os, sys, subprocess, importlib.util

# Kaggle often ships SHAP; install only missing dependencies.
missing = [name for name in ("shap", "lime") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing packages:", missing)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Package installation failed. In Kaggle, enable Internet in notebook settings, or attach offline wheels for: " + ", ".join(missing)) from error
print("Required explainability packages available.")

# %% [cell 3]
import io, re, warnings, json, zipfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import shap
from lime.lime_tabular import LimeTabularExplainer
from IPython.display import display, Markdown

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
OUTPUT_DIR = ROOT / 'emi_explainability_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GITHUB_REPOSITORY_URL = 'https://github.com/Yash-Bharvada/RSA-SetB-EMI-Explainability'
print('SHAP:', shap.__version__, '| Output directory:', OUTPUT_DIR)


# %% [cell 5]
GERMAN_COLUMNS = [
    'checking_status','duration','credit_history','purpose','credit_amount',
    'savings_status','employment','installment_commitment','personal_status',
    'other_parties','residence_since','property_magnitude','age',
    'other_payment_plans','housing','existing_credits','job',
    'num_dependents','own_telephone','foreign_worker','class'
]

search_roots = [Path('/kaggle/input'), Path('/kaggle/working'), Path.cwd(), Path('/mnt/data')]
candidates = []
for search_root in search_roots:
    if search_root.exists():
        for pattern in ('*german*credit*.csv','*German*Credit*.csv','german.data','*german*.csv'):
            candidates.extend(search_root.rglob(pattern))
candidates = sorted(set(x for x in candidates if x.is_file()), key=lambda x: (x.name.lower() != 'german_credit.csv', len(str(x))))
if not candidates:
    raise FileNotFoundError('German Credit dataset not found. Kaggle: Add Input -> attach german_credit.csv or german.data, then rerun this cell. Dataset source: UCI Statlog (German Credit Data).')
DATASET_PATH = candidates[0]
print('Using actual dataset:', DATASET_PATH)

def read_german_credit(path):
    if path.suffix.lower() == '.data':
        result = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    else:
        result = pd.read_csv(path)
        # Some CSV exports include a numeric first line instead of proper headers.
        if result.shape[1] == 21 and all(str(c).strip().isdigit() for c in result.columns):
            result = pd.read_csv(path, header=None)
        if result.shape[1] == 1:
            result = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    if result.shape[1] == 21:
        if all(isinstance(c, (int, np.integer)) for c in result.columns) or all(str(c).startswith('Unnamed:') for c in result.columns):
            result.columns = GERMAN_COLUMNS
        elif 'class' not in [str(c).strip().lower() for c in result.columns] and str(result.columns[0]).strip().isdigit():
            result.columns = GERMAN_COLUMNS
    # Standardize frequently-used Kaggle/OpenML names.
    norm = lambda name: re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')
    result = result.rename(columns={c: norm(c) for c in result.columns})
    aliases = {'risk':'class','target':'class','credit_risk':'class','creditability':'class',
               'creditability_':'class','checking_account':'checking_status',
               'checking_account_status':'checking_status', 'credit_duration':'duration',
               'amount':'credit_amount','savings_account':'savings_status',
               'personal_status_and_sex':'personal_status', 'foreign_worker_':'foreign_worker'}
    result = result.rename(columns={c: aliases.get(c,c) for c in result.columns})
    if 'class' not in result.columns and len(result.columns) == 21:
        for possible in ('credit_risk','credit_rating','response','default'):
            if possible in result.columns:
                result = result.rename(columns={possible:'class'})
                break
    if 'class' not in result.columns:
        raise ValueError('Could not determine the target column. Expected class/risk/target/creditability. Columns: ' + str(result.columns.tolist()))
    return result

df = read_german_credit(DATASET_PATH)
if df.columns.duplicated().any():
    raise ValueError('Duplicate column names after normalization; inspect the source CSV.')
if df.isna().any().any():
    raise ValueError('Input data contain missing values. Use the original complete German Credit dataset for reproducible results.')
print('Rows/columns:', df.shape)
display(df.head(5))
print('Original class counts:')
print(df['class'].value_counts(dropna=False))

# %% [cell 6]
def target_bad_one(series):
    raw = series.astype(str).str.strip().str.lower()
    labels = set(raw.unique())
    if labels == {'1','2'}:  # UCI Statlog: 1 good; 2 bad
        return raw.map({'1':0,'2':1}).astype(int)
    if labels == {'good','bad'}:
        return raw.map({'good':0,'bad':1}).astype(int)
    if labels == {'0','1'}:  # Common 0=good, 1=bad CSV conventions
        return raw.astype(int)
    if labels == {'0','1','2'}:
        raise ValueError('Unexpected three-class target; please inspect the dataset.')
    raise ValueError(f'Unsupported target labels {labels}. Inspect labels to ensure 1 means bad; do not guess mapping.')

y = target_bad_one(df['class'])
X = df.drop(columns=['class']).copy()
assert len(X) == len(y) and y.nunique() == 2
assert X.shape[1] == 20, f'Expected 20 German Credit attributes, got {X.shape[1]}'

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
cat_cols = X_train_raw.select_dtypes(include=['object','category','string']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()
if cat_cols:
    X_train.loc[:, cat_cols] = encoder.fit_transform(X_train_raw[cat_cols].astype(str))
    X_test.loc[:, cat_cols] = encoder.transform(X_test_raw[cat_cols].astype(str))
for c in X.columns:
    X_train[c] = pd.to_numeric(X_train[c], errors='raise').astype(float)
    X_test[c] = pd.to_numeric(X_test[c], errors='raise').astype(float)
X_train, X_test = X_train[X.columns], X_test[X.columns]
assert X_train.shape[0] == 800 and X_test.shape[0] == 200 if len(df) == 1000 else len(X_train) + len(X_test) == len(df)
print('Train:', X_train.shape, 'Test:', X_test.shape)
print('Encoded categorical features:', len(cat_cols), '| Numeric features:', len(num_cols))
print('Positive = BAD (1); train bad proportion =', round(y_train.mean(),3), 'test =', round(y_test.mean(),3))


# %% [cell 8]
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
p_bad = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, p_bad)
print('Model: RandomForest (200 trees, depth 8)')
print(f'Test accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}')
print('\nClassification report (1 = BAD):\n', classification_report(y_test, y_pred, target_names=['good','bad'], zero_division=0))
fig, ax = plt.subplots(figsize=(5.4,4.5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['good','bad'], cmap='Blues', ax=ax)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'confusion_matrix.png', dpi=170, bbox_inches='tight'); plt.show()
fig, ax = plt.subplots(figsize=(5.7,4.5))
RocCurveDisplay.from_predictions(y_test, p_bad, pos_label=1, ax=ax)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'roc_curve.png', dpi=170, bbox_inches='tight'); plt.show()


# %% [cell 10]
explainer = shap.TreeExplainer(model)
raw_shap = explainer.shap_values(X_test)

def bad_class_shap(values, n_samples, n_features):
    if isinstance(values, list):
        values = values[list(model.classes_).index(1)]
    arr = np.asarray(values.values if hasattr(values, 'values') else values)
    if arr.ndim == 3:
        if arr.shape == (n_samples, n_features, 2):
            arr = arr[:,:,list(model.classes_).index(1)]
        elif arr.shape == (2, n_samples, n_features):
            arr = arr[list(model.classes_).index(1)]
        else:
            raise ValueError(f'Unexpected 3D SHAP shape {arr.shape}')
    if arr.shape != (n_samples, n_features):
        raise ValueError(f'Unexpected SHAP shape {arr.shape}')
    return np.asarray(arr, dtype=float)

shap_bad = bad_class_shap(raw_shap, len(X_test), X_test.shape[1])
importance = pd.Series(np.abs(shap_bad).mean(axis=0), index=X.columns).sort_values(ascending=False)
top5 = importance.head(5)
print('GLOBAL FEATURE IMPORTANCE (mean |SHAP|, class = bad)')
print(top5.rename('mean_abs_shap').to_string(float_format=lambda x: f'{x:.5f}'))
top5.rename('mean_abs_shap').to_csv(OUTPUT_DIR/'global_top5_shap.csv', header=True)

# Use the consistent, version-resistant summary bar chart.
plt.figure(figsize=(9,7))
shap.summary_plot(shap_bad, X_test, feature_names=X.columns.tolist(), plot_type='bar', show=False, max_display=20)
plt.title('Global SHAP importance — P(bad)', pad=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'shap_summary.png', dpi=180, bbox_inches='tight')
plt.show()


# %% [cell 12]
rejected_positions = np.flatnonzero(y_pred == 1)
if len(rejected_positions) == 0:
    raise RuntimeError('This trained model predicted no bad applicants in the test set. Cannot honestly provide a rejected-applicant explanation.')
app_pos = int(rejected_positions[0])
applicant = X_test.iloc[app_pos]
applicant_original = X_test_raw.iloc[app_pos]
app_probability = float(p_bad[app_pos])
app_shap = shap_bad[app_pos]
app_reason_indices = np.argsort(np.abs(app_shap))[::-1][:3]
shap_top3_features = [str(X.columns[i]) for i in app_reason_indices]

# A SHAP waterfall Explanation requires a scalar expected value and 1D feature contribution vector.
expected = np.asarray(explainer.expected_value, dtype=float)
if expected.ndim == 0:
    baseline = float(expected)
else:
    baseline = float(expected.reshape(-1)[list(model.classes_).index(1)])
local_exp = shap.Explanation(values=app_shap, base_values=baseline,
                            data=applicant.to_numpy(dtype=float), feature_names=X.columns.tolist())
print(f'Applicant test position {app_pos}, original index {X_test.index[app_pos]}: P(bad) = {app_probability:.4f} -> REJECTED')
print('SHAP base + sum(contributions):', round(baseline+float(app_shap.sum()),5))
for rank, i in enumerate(app_reason_indices, 1):
    feature = X.columns[i]
    print(f'{rank}. {feature} = {applicant_original[feature]} | SHAP = {app_shap[i]:+.5f} | ' + ('increases risk' if app_shap[i]>0 else 'decreases risk'))
shap.plots.waterfall(local_exp, max_display=12, show=False)
plt.gcf().set_size_inches(10,7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'shap_waterfall.png', dpi=180, bbox_inches='tight')
plt.show()
assert np.isclose(baseline+app_shap.sum(),app_probability,atol=2e-3), 'SHAP and model probability mismatch; check SHAP version.'


# %% [cell 14]
lime_explainer = LimeTabularExplainer(
    training_data=X_train.to_numpy(dtype=float),
    feature_names=X.columns.tolist(),
    class_names=['good','bad'],
    mode='classification',
    discretize_continuous=False,
    categorical_features=[X.columns.get_loc(c) for c in cat_cols],
    categorical_names={X.columns.get_loc(c): [str(v) for v in encoder.categories_[j]] for j,c in enumerate(cat_cols)},
    random_state=RANDOM_STATE
)
lime_exp = lime_explainer.explain_instance(
    data_row=applicant.to_numpy(dtype=float),
    predict_fn=model.predict_proba,
    num_features=5,
    labels=(1,),
    num_samples=3000
)
lime_pairs = lime_exp.as_list(label=1)
lime_feature_ids = [i for i,_ in lime_exp.as_map()[1]]
lime_top3_features = [str(X.columns[i]) for i in lime_feature_ids[:3]]
agreement = len(set(shap_top3_features) & set(lime_top3_features))
print('LIME top 5 (bad class):')
for feature_text, weight in lime_pairs:
    print(f'  {feature_text}: {weight:+.5f}')
print('\nSHAP top 3:', shap_top3_features)
print('LIME top 3:', lime_top3_features)
print(f'Top-3 agreement (SHAP vs LIME): {agreement} / 3')
fig = lime_exp.as_pyplot_figure(label=1)
fig.set_size_inches(9,5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/'lime_explanation.png', dpi=180, bbox_inches='tight')
plt.show()
with open(OUTPUT_DIR/'lime_explanation.html', 'w', encoding='utf-8') as f:
    f.write(lime_exp.as_html())


# %% [cell 16]
SENSITIVE = {'age','personal_status','foreign_worker'}
shap_sensitive_top3 = sorted(SENSITIVE.intersection(shap_top3_features))
lime_sensitive_top5 = sorted(SENSITIVE.intersection([str(X.columns[i]) for i in lime_feature_ids]))
flagged = sorted(set(shap_sensitive_top3) | set(lime_sensitive_top5))

# Restrict stated rejection reasons to positive (risk-increasing) SHAP values.
risk_positive_indices = [i for i in np.argsort(app_shap)[::-1] if app_shap[i] > 0 and X.columns[i] not in SENSITIVE]
selected_reasons = risk_positive_indices[:3]

FRIENDLY = {
    'checking_status': 'your checking-account status',
    'duration': 'the requested repayment duration',
    'credit_amount': 'the requested credit amount',
    'savings_status': 'your savings-account status',
    'credit_history': 'your credit history',
    'installment_commitment': 'the proposed installment burden',
    'existing_credits': 'the number of existing credits',
    'employment': 'the employment-history category',
    'purpose': 'the stated loan purpose',
    'other_payment_plans': 'your existing payment-plan arrangements',
    'residence_since': 'the time at your current residence',
    'housing': 'the recorded housing category',
    'property_magnitude': 'the recorded property category',
    'other_parties': 'the recorded co-applicant or guarantor category',
    'job': 'the recorded employment category',
    'num_dependents': 'the number of dependents',
    'own_telephone': 'the recorded telephone status'
}
reason_phrases = [f"{FRIENDLY.get(X.columns[i], X.columns[i].replace('_',' '))} ({applicant_original[X.columns[i]]})" for i in selected_reasons]
if reason_phrases:
    if len(reason_phrases) == 1:
        message_end = reason_phrases[0]
    else:
        message_end = ', '.join(reason_phrases[:-1]) + ' and ' + reason_phrases[-1]
    rejection_message = 'Your EMI application was not approved by the automated model. The main non-sensitive factors increasing its estimated credit risk were ' + message_end + '. You may request a human review of the decision.'
else:
    rejection_message = 'Your EMI application was not approved by the automated model. We cannot reliably provide a non-sensitive risk-increasing reason from this local explanation; please request a human review.'

print('Sensitive among SHAP top 3:', shap_sensitive_top3 or 'None')
print('Sensitive among LIME top 5:', lime_sensitive_top5 or 'None')
print('OVERALL FLAG:', 'REVIEW' if flagged else 'No sensitive feature in the inspected top explanations')
print('\nPLAIN-LANGUAGE REJECTION MESSAGE:\n' + rejection_message)
print('\nAudit note: SHAP and LIME explain model behavior, not causal evidence or a legal fairness determination.')
(OUTPUT_DIR/'rejection_message.txt').write_text(rejection_message+'\n', encoding='utf-8')


# %% [cell 18]
def readable_top3():
    return '; '.join(f"{X.columns[i]}={applicant_original[X.columns[i]]} (SHAP {app_shap[i]:+.4f})" for i in app_reason_indices)

qa = {
    'Q1': f'Accuracy = {accuracy:.4f}; ROC-AUC = {roc_auc:.4f}. Top 5 global features: ' + ', '.join(f'{name} ({val:.4f})' for name,val in top5.items()) + '. These show the strongest overall model contributions to variation in the estimated probability of bad credit; they are not causal proof.',
    'Q2': 'Top 3 SHAP reasons for the selected rejected applicant: ' + readable_top3() + '. A positive SHAP value pushes P(bad) above its baseline; a negative value pushes it below the baseline.',
    'Q3': f'SHAP/LIME top-3 feature agreement = {agreement}/3. SHAP top 3: {shap_top3_features}. LIME top 3: {lime_top3_features}. They can differ because SHAP attributes the forest prediction using a background distribution, whereas LIME estimates a local surrogate from sampled perturbations.',
    'Q4': 'Rejection message: ' + rejection_message + ' Sensitive among SHAP top 3: ' + (', '.join(shap_sensitive_top3) if shap_sensitive_top3 else 'none') + '. Sensitive among LIME top 5: ' + (', '.join(lime_sensitive_top5) if lime_sensitive_top5 else 'none') + '. ' + ('Flag for fairness/compliance review; check performance and disparities across groups and obtain human oversight.' if flagged else 'Continue monitoring for disparate impact and periodically audit the model.')
}
for q,a in qa.items():
    display(Markdown(f'**{q}.** {a}'))
(OUTPUT_DIR/'technical_answers_Q1_Q4.txt').write_text('\n\n'.join(f'{q}. {a}' for q,a in qa.items()), encoding='utf-8')
results = {
    'dataset_path': str(DATASET_PATH), 'dataset_shape': list(df.shape),
    'random_state':42, 'train_size':len(X_train), 'test_size':len(X_test),
    'accuracy':float(accuracy), 'roc_auc':float(roc_auc),
    'global_top5':{str(k):float(v) for k,v in top5.items()},
    'test_position':app_pos,'original_test_index':str(X_test.index[app_pos]),
    'predicted_p_bad':app_probability,
    'shap_top3':shap_top3_features, 'lime_top3':lime_top3_features,
    'top3_agreement':agreement, 'sensitive_flags':flagged, 'rejection_message':rejection_message,
    'github_repository_url':GITHUB_REPOSITORY_URL
}
(OUTPUT_DIR/'observed_results.json').write_text(json.dumps(results,indent=2), encoding='utf-8')


# %% [cell 20]
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

def report_page(pdf, heading, paragraphs, images=()):
    fig = plt.figure(figsize=(8.27,11.69))
    fig.patch.set_facecolor('white')
    fig.text(.08,.94,heading,fontsize=16,weight='bold',va='top')
    y=.89
    for paragraph in paragraphs:
        lines=[]
        for line in str(paragraph).split('\n'):
            lines.extend(textwrap.wrap(line, width=96, break_long_words=False) or [''])
        line_height=.016
        for line in lines:
            if y < .14: break
            fig.text(.08,y,line,fontsize=9.2,va='top',family='sans-serif')
            y-=line_height
        y-=.015
    for image_path in images:
        if not Path(image_path).exists(): continue
        if y < .30: break
        ax=fig.add_axes([.11,max(.1,y-.34),.78,.32]); ax.imshow(plt.imread(image_path)); ax.axis('off')
        y-=.37
    fig.text(.08,.045,'RSA AIML4001 | External Practical Set B | Generated from observed notebook results',fontsize=8,color='gray')
    pdf.savefig(fig,bbox_inches=None); plt.close(fig)

pdf_path=OUTPUT_DIR/'RSA_SetB_Practical_Report.pdf'
with PdfPages(pdf_path) as pdf:
    report_page(pdf,'EMI Loan Explainability — Practical Report',[
        'Subject: Responsible and Safe AI (AIML4001) | Set B | 25/11/2026',
        'GitHub repository: '+GITHUB_REPOSITORY_URL,
        'Aim: Explain the global and applicant-specific reasons behind automated EMI loan rejections using SHAP and LIME while auditing sensitive attributes.',
        'Dataset: UCI Statlog German Credit (or the explicitly attached equivalent): '+DATASET_PATH.name+'. Dataset dimensions: '+str(df.shape)+'.',
        'Algorithm: 1) Load and validate data, map bad to 1. 2) Stratified 80:20 split (seed 42), fit OrdinalEncoder on training categories. 3) Fit 200-tree Random Forest with depth 8. 4) Evaluate accuracy and ROC-AUC. 5) Obtain TreeExplainer SHAP global and local values. 6) Explain the rejected applicant with LIME. 7) Compare features, check sensitive attributes, and generate a customer-facing message.',
        'Program: Complete, executable Python source code is in RSA_SetB_EMI_Explainability_Kaggle.ipynb (also included with deliverables). Core Python libraries: pandas, scikit-learn, SHAP, LIME, matplotlib.',
        f'Observed output: Train = {len(X_train)}, test = {len(X_test)}, accuracy = {accuracy:.4f}, ROC-AUC = {roc_auc:.4f}. Model = RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42).',
        'Global top 5: '+', '.join(f'{k} ({v:.4f})' for k,v in top5.items()),
    ])
    report_page(pdf,'Global explanation and model evaluation',[
        'The chart shows mean absolute SHAP contribution (class bad=1). Important features describe the model, not causal effects.'
    ],images=[OUTPUT_DIR/'shap_summary.png'])
    report_page(pdf,'Local SHAP explanation (rejected applicant)',[
        f'Test position: {app_pos}; model P(bad) = {app_probability:.4f}; predicted class: bad (rejected).',
        'Top 3 SHAP: '+readable_top3(),
        'Positive SHAP -> pushes predicted bad probability upward; negative SHAP -> downward.'
    ],images=[OUTPUT_DIR/'shap_waterfall.png'])
    report_page(pdf,'LIME vs SHAP comparison',[
        'SHAP top 3: '+', '.join(shap_top3_features),
        'LIME top 3: '+', '.join(lime_top3_features),
        f'Top-three feature overlap: {agreement}/3. Methods can differ because they construct explanations differently.'
    ],images=[OUTPUT_DIR/'lime_explanation.png'])
    report_page(pdf,'Rejection communication and technical answers',[
        'Plain-language rejection: '+rejection_message,
        'Sensitivity audit: '+(', '.join(flagged)+' -> REVIEW' if flagged else 'No sensitive attribute in inspected top reasons.'),
        *[f'{q}: {answer}' for q,answer in qa.items()],
        'Conclusion: The Random Forest can be globally and locally explained with SHAP and LIME, but explanations alone are insufficient to establish fairness. Sensitive-feature use requires careful compliance review, validation, human oversight and an appeal path.'
    ])
print('Report generated:', pdf_path)


# %% [cell 22]
zip_path = ROOT/'RSA_SetB_EMI_Outputs.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for file in sorted(OUTPUT_DIR.rglob('*')):
        if file.is_file():
            z.write(file,arcname=f'emi_explainability_outputs/{file.relative_to(OUTPUT_DIR)}')
print('All files in:', OUTPUT_DIR)
print('Download PDF:', pdf_path)
print('Download ZIP:', zip_path)
print('Remember: create a GitHub repository and upload this .ipynb plus relevant code and figures.')
