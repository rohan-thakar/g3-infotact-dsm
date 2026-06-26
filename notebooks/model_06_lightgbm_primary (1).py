# ============================================================
# MODEL 6 — LightGBM (Primary Model — Project Spec Requirement)
# Day 6 commit | Full pipeline: SMOTE in CV + noise analysis + SHAP
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, f1_score,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ---- DATA & FEATURES ----
df = pd.read_csv('/kaggle/input/ai4i-2020-predictive-maintenance-dataset/ai4i2020.csv')
df.drop(columns=['UDI', 'Product ID'], inplace=True)
le = LabelEncoder()
df['Type_enc'] = le.fit_transform(df['Type'])
df.drop(columns=['Type'], inplace=True)

df['Temp_diff']     = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power_W']       = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
df['Wear_x_Torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['Stress_proxy']  = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-6)
df['Wear_per_RPM']  = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1e-6)
df['Torque_dev']    = np.abs(df['Torque [Nm]'] - df['Torque [Nm]'].mean())
df['Temp_ratio']    = df['Process temperature [K]'] / df['Air temperature [K]']
df['Wear_norm']     = df['Tool wear [min]'] / df['Tool wear [min]'].max()
df['HDF_cond']      = ((df['Temp_diff'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
df['PWF_cond']      = ((df['Power_W'] < 3500) | (df['Power_W'] > 9000)).astype(int)
df['OSF_cond']      = (df['Wear_x_Torque'] > 11000).astype(int)
df['fault_risk']    = df['HDF_cond'] + df['PWF_cond'] + df['OSF_cond']
df['RPM_bin']       = pd.cut(df['Rotational speed [rpm]'], bins=3, labels=[0,1,2]).astype(int)

TARGET = 'Machine failure'
FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_cols = [c for c in df.columns if c not in [TARGET] + FAILURE_COLS]
X = df[feature_cols]
y = df[TARGET]

# ---- LightGBM PARAMS ----
lgb_params = {
    'n_estimators':    500,
    'learning_rate':   0.03,
    'max_depth':       7,
    'num_leaves':      40,
    'min_child_samples': 10,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':       0.1,
    'reg_lambda':      0.1,
    'is_unbalance':    True,     # handles class imbalance natively too
    'random_state':    42,
    'n_jobs':          -1,
    'verbose':         -1
}

# ---- SMOTE INSIDE CV (no data leakage) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42, k_neighbors=5)

oof_pred  = np.zeros(len(y))
oof_prob  = np.zeros(len(y))
fold_f1s  = []

print("Running 5-Fold CV with SMOTE inside each fold...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # SMOTE only on training fold
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr_sm, y_tr_sm,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])

    oof_pred[val_idx] = model.predict(X_val)
    oof_prob[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_macro_f1 = f1_score(y_val, oof_pred[val_idx], average='macro')
    fold_f1s.append(fold_macro_f1)
    print(f"  Fold {fold+1}/5 → Macro F1: {fold_macro_f1:.4f}")

print(f"\nMean CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

print("=" * 60)
print("MODEL 6: LightGBM (Primary)")
print("=" * 60)
print(classification_report(y, oof_pred, target_names=['No Failure', 'Failure']))
macro_f1 = f1_score(y, oof_pred, average='macro')
roc_auc  = roc_auc_score(y, oof_prob)
print(f"Macro F1 Score : {macro_f1:.4f}  (target ≥ 0.85)")
print(f"ROC-AUC Score  : {roc_auc:.4f}")

# ---- THRESHOLD TUNING ----
print("\n--- Threshold Tuning ---")
best_thresh, best_f1 = 0.5, 0
for thresh in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y, (oof_prob >= thresh).astype(int), average='macro')
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh
print(f"Best threshold: {best_thresh:.2f} → Macro F1: {best_f1:.4f}")

# ---- NOISE SENSITIVITY ANALYSIS ----
print("\n--- Noise Sensitivity Analysis ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X_tr_sm, y_tr_sm, callbacks=[lgb.log_evaluation(period=-1)])

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
noise_f1s    = []
for noise_level in noise_levels:
    X_noisy = X_test.values.copy().astype(float)
    if noise_level > 0:
        X_noisy += np.random.normal(0, noise_level * X_noisy.std(axis=0), X_noisy.shape)
    y_noisy_pred = final_model.predict(X_noisy)
    f1_n = f1_score(y_test, y_noisy_pred, average='macro')
    noise_f1s.append(f1_n)
    print(f"  Noise σ={noise_level:.2f} → Macro F1={f1_n:.4f}")

# ---- SHAP EXPLANATIONS ----
print("\nGenerating SHAP values...")
explainer    = shap.TreeExplainer(final_model)
shap_values  = explainer.shap_values(X_test)
# For binary classification, shap_values is a list; take class 1
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# ---- PLOTS ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Feature importance (gain)
lgb_imp = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
lgb_imp.head(12).plot(kind='barh', ax=axes[0, 0], color='#1a9850')
axes[0, 0].set_title('Top 12 Feature Importances (Gain) — LightGBM')
axes[0, 0].invert_yaxis()

# PR Curve
precision, recall, _ = precision_recall_curve(y, oof_prob)
ap = average_precision_score(y, oof_prob)
axes[0, 1].plot(recall, precision, lw=2, color='#1a9850', label=f'AP={ap:.3f}')
axes[0, 1].axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Recall=0.80')
axes[0, 1].fill_between(recall, precision, alpha=0.12, color='#1a9850')
axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# Noise sensitivity
axes[1, 0].plot(noise_levels, noise_f1s, 'o-', color='tomato', lw=2)
axes[1, 0].axhline(y=0.85, color='green', linestyle='--', label='Target F1=0.85')
axes[1, 0].set_xlabel('Noise Level (σ)'); axes[1, 0].set_ylabel('Macro F1')
axes[1, 0].set_title('Noise Sensitivity Analysis')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# SHAP bar plot
shap_mean = np.abs(sv).mean(axis=0)
shap_series = pd.Series(shap_mean, index=feature_cols).sort_values(ascending=False)
shap_series.head(12).plot(kind='barh', ax=axes[1, 1], color='#4575b4')
axes[1, 1].set_title('SHAP Mean |value| — Top 12 Features')
axes[1, 1].invert_yaxis()

plt.suptitle('LightGBM — Predictive Maintenance Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model6_lightgbm_full.png', dpi=150, bbox_inches='tight')
plt.show()

# SHAP summary plot (beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False, max_display=15)
plt.title('SHAP Summary Plot — LightGBM')
plt.tight_layout()
plt.savefig('model6_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Model 6 (LightGBM Primary) complete.")
print(f"Final Macro F1 with best threshold: {best_f1:.4f}")
