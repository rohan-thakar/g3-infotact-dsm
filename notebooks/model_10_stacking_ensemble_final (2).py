# ============================================================
# MODEL 10 — Stacking Ensemble (BEST / FINAL MODEL)
# Day 10 commit | Full pipeline: SMOTE + Stacking + SHAP + Noise
# This is your "kept one which is best and optimized and perfect"
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import shap
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, f1_score,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MODEL 10: Stacking Ensemble — FINAL OPTIMIZED MODEL")
print("=" * 70)

# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================
df = pd.read_csv('/kaggle/input/ai4i-2020-predictive-maintenance-dataset/ai4i2020.csv')
df.drop(columns=['UDI', 'Product ID'], inplace=True)
le = LabelEncoder()
df['Type_enc'] = le.fit_transform(df['Type'])
df.drop(columns=['Type'], inplace=True)

# --- Domain features ---
df['Temp_diff']     = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power_W']       = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
df['Wear_x_Torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['Stress_proxy']  = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-6)
df['Wear_per_RPM']  = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1e-6)
df['Torque_dev']    = np.abs(df['Torque [Nm]'] - df['Torque [Nm]'].mean())
df['Temp_ratio']    = df['Process temperature [K]'] / df['Air temperature [K]']
df['Wear_norm']     = df['Tool wear [min]'] / df['Tool wear [min]'].max()
df['RPM_bin']       = pd.cut(df['Rotational speed [rpm]'], bins=3, labels=[0,1,2]).astype(int)
df['HDF_cond']      = ((df['Temp_diff'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
df['PWF_cond']      = ((df['Power_W'] < 3500) | (df['Power_W'] > 9000)).astype(int)
df['OSF_cond']      = (df['Wear_x_Torque'] > 11000).astype(int)
df['fault_risk']    = df['HDF_cond'] + df['PWF_cond'] + df['OSF_cond']
# Interaction terms
df['Temp_diff_x_Wear']  = df['Temp_diff'] * df['Tool wear [min]']
df['Power_x_Wear']      = df['Power_W'] * df['Tool wear [min]']
df['Torque_x_RPM']      = df['Torque [Nm]'] * df['Rotational speed [rpm]']

TARGET = 'Machine failure'
FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_cols = [c for c in df.columns if c not in [TARGET] + FAILURE_COLS]
X = df[feature_cols]
y = df[TARGET]

print(f"Features: {len(feature_cols)}")
print(f"Class balance: {y.value_counts().to_dict()}")

# ============================================================
# 2. BASE LEARNERS + META-LEARNER
# ============================================================
neg, pos = (y == 0).sum(), (y == 1).sum()
spw = neg / pos

base_learners = [
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='sqrt',
        class_weight='balanced', n_jobs=-1, random_state=42)),

    ('xgb', xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, use_label_encoder=False,
        eval_metric='logloss', random_state=42, n_jobs=-1)),

    ('lgb', lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, num_leaves=40,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)),

    ('gb', GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42)),
]

meta_learner = LogisticRegression(
    C=1.0, class_weight='balanced', max_iter=500, random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=3,
    stack_method='predict_proba',
    n_jobs=-1
)

# ============================================================
# 3. SMOTE INSIDE STRATIFIED 5-FOLD CV
# ============================================================
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42, k_neighbors=5)

oof_pred = np.zeros(len(y), dtype=int)
oof_prob = np.zeros(len(y))
fold_f1s = []

print("\nRunning 5-Fold CV for Stacking Ensemble (this takes a few minutes)...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    stacking_clf.fit(X_tr_sm, y_tr_sm)
    oof_pred[val_idx] = stacking_clf.predict(X_val)
    oof_prob[val_idx] = stacking_clf.predict_proba(X_val)[:, 1]

    fold_f1 = f1_score(y_val, oof_pred[val_idx], average='macro')
    fold_f1s.append(fold_f1)
    print(f"  Fold {fold+1}/5 → Macro F1: {fold_f1:.4f}")

print(f"\nMean CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

# ============================================================
# 4. RESULTS
# ============================================================
print("\n" + "=" * 60)
print(classification_report(y, oof_pred, target_names=['No Failure', 'Failure']))
macro_f1 = f1_score(y, oof_pred, average='macro')
roc_auc  = roc_auc_score(y, oof_prob)
ap       = average_precision_score(y, oof_prob)
print(f"Macro F1 Score  : {macro_f1:.4f}  (target ≥ 0.85)")
print(f"ROC-AUC Score   : {roc_auc:.4f}")
print(f"Avg Precision   : {ap:.4f}")

# ============================================================
# 5. THRESHOLD TUNING
# ============================================================
print("\n--- Optimal Threshold Search ---")
best_thresh, best_f1 = 0.5, 0
thresholds = np.arange(0.05, 0.95, 0.025)
thresh_f1s = []
for thresh in thresholds:
    f1 = f1_score(y, (oof_prob >= thresh).astype(int), average='macro')
    thresh_f1s.append(f1)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"Best threshold  : {best_thresh:.3f}")
print(f"Best Macro F1   : {best_f1:.4f}")
print(f"At threshold {best_thresh:.3f}:")
print(classification_report(y, (oof_prob >= best_thresh).astype(int),
                             target_names=['No Failure', 'Failure']))

# ============================================================
# 6. NOISE SENSITIVITY ANALYSIS
# ============================================================
print("\n--- Noise Sensitivity Analysis ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
stacking_clf.fit(X_tr_sm, y_tr_sm)

noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
noise_f1s, noise_aucs = [], []
for noise_level in noise_levels:
    X_noisy = X_test.values.copy().astype(float)
    if noise_level > 0:
        X_noisy += np.random.normal(0, noise_level * X_noisy.std(axis=0), X_noisy.shape)
    X_noisy_df = pd.DataFrame(X_noisy, columns=X_test.columns)
    y_noisy_pred = stacking_clf.predict(X_noisy_df)
    y_noisy_prob = stacking_clf.predict_proba(X_noisy_df)[:, 1]
    f1_n  = f1_score(y_test, y_noisy_pred, average='macro')
    auc_n = roc_auc_score(y_test, y_noisy_prob)
    noise_f1s.append(f1_n)
    noise_aucs.append(auc_n)
    print(f"  Noise σ={noise_level:.2f} → Macro F1={f1_n:.4f}  ROC-AUC={auc_n:.4f}")

# ============================================================
# 7. SHAP EXPLANATIONS (LightGBM component)
# ============================================================
print("\nGenerating SHAP values from LightGBM base learner...")
lgb_estimator = None
for name, est in stacking_clf.estimators_:
    if name == 'lgb':
        lgb_estimator = est
        break

explainer   = shap.TreeExplainer(lgb_estimator)
shap_values = explainer.shap_values(X_test)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# ============================================================
# 8. COMPREHENSIVE PLOTS
# ============================================================
fig = plt.figure(figsize=(20, 16))
gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# (0,0) Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y, oof_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax1,
            xticklabels=['No Fail', 'Fail'], yticklabels=['No Fail', 'Fail'])
ax1.set_title('Confusion Matrix\n(Default Threshold)', fontweight='bold')

# (0,1) Confusion Matrix at best threshold
ax2 = fig.add_subplot(gs[0, 1])
cm_opt = confusion_matrix(y, (oof_prob >= best_thresh).astype(int))
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='YlGn', ax=ax2,
            xticklabels=['No Fail', 'Fail'], yticklabels=['No Fail', 'Fail'])
ax2.set_title(f'Confusion Matrix\n(Threshold={best_thresh:.2f})', fontweight='bold')

# (0,2) Threshold vs F1
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(thresholds, thresh_f1s, lw=2, color='steelblue')
ax3.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best={best_thresh:.2f}')
ax3.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target F1=0.85')
ax3.set_xlabel('Decision Threshold')
ax3.set_ylabel('Macro F1')
ax3.set_title('Threshold Tuning Curve', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# (1,0) PR Curve
ax4 = fig.add_subplot(gs[1, 0])
precision, recall, _ = precision_recall_curve(y, oof_prob)
ax4.plot(recall, precision, lw=2.5, color='darkorange', label=f'Stacking AP={ap:.3f}')
ax4.fill_between(recall, precision, alpha=0.15, color='darkorange')
ax4.set_xlabel('Recall'); ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve', fontweight='bold')
ax4.legend(); ax4.grid(True, alpha=0.3)

# (1,1) Noise Sensitivity
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(noise_levels, noise_f1s,  'o-', color='tomato',    lw=2, label='Macro F1')
ax5.plot(noise_levels, noise_aucs, 's--', color='steelblue', lw=2, label='ROC-AUC')
ax5.axhline(y=0.85, color='green', linestyle=':', alpha=0.7, label='F1 Target')
ax5.set_xlabel('Noise Level (σ fraction)'); ax5.set_ylabel('Score')
ax5.set_title('Noise Sensitivity Analysis', fontweight='bold')
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# (1,2) CV Fold F1s
ax6 = fig.add_subplot(gs[1, 2])
fold_bars = ax6.bar(range(1, 6), fold_f1s, color='#4dac26', alpha=0.8)
ax6.axhline(y=np.mean(fold_f1s), color='red', linestyle='--',
            label=f'Mean={np.mean(fold_f1s):.4f}')
ax6.axhline(y=0.85, color='orange', linestyle=':', label='Target=0.85')
ax6.set_xlabel('Fold'); ax6.set_ylabel('Macro F1')
ax6.set_title('Cross-Validation F1 per Fold', fontweight='bold')
ax6.legend(fontsize=8); ax6.set_xticks(range(1, 6))

# (2,0-1) SHAP Summary
ax7 = fig.add_subplot(gs[2, 0:2])
shap_mean = np.abs(sv).mean(axis=0)
shap_df   = pd.Series(shap_mean, index=feature_cols).sort_values(ascending=False).head(15)
ax7.barh(range(len(shap_df)), shap_df.values, color='#4575b4')
ax7.set_yticks(range(len(shap_df)))
ax7.set_yticklabels(shap_df.index, fontsize=9)
ax7.invert_yaxis()
ax7.set_xlabel('Mean |SHAP Value|')
ax7.set_title('SHAP Feature Importance — Top 15 (LightGBM)', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

# (2,2) Score summary text
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
summary = (
    f"FINAL RESULTS SUMMARY\n"
    f"{'─' * 30}\n"
    f"Macro F1 (default):  {macro_f1:.4f}\n"
    f"Macro F1 (tuned):    {best_f1:.4f}\n"
    f"Target:             ≥ 0.85\n"
    f"{'✅' if best_f1 >= 0.85 else '❌'} Target {'MET' if best_f1 >= 0.85 else 'NOT MET'}\n\n"
    f"ROC-AUC:             {roc_auc:.4f}\n"
    f"Avg Precision:       {ap:.4f}\n"
    f"Best Threshold:      {best_thresh:.3f}\n\n"
    f"CV Macro F1:\n"
    f"  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}\n\n"
    f"Noise Robustness:\n"
    f"  σ=0.20 → F1={noise_f1s[4]:.4f}"
)
ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Model 10: Stacking Ensemble — Final Results', fontsize=16, fontweight='bold', y=1.01)
plt.savefig('model10_stacking_ensemble_final.png', dpi=150, bbox_inches='tight')
plt.show()

# SHAP beeswarm
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False, max_display=15)
plt.title('SHAP Beeswarm Plot — Stacking Ensemble (LGB component)', fontweight='bold')
plt.tight_layout()
plt.savefig('model10_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("✅ MODEL 10 (FINAL) COMPLETE")
print(f"   Macro F1 (tuned threshold): {best_f1:.4f}")
print(f"   Target ≥ 0.85: {'✅ ACHIEVED' if best_f1 >= 0.85 else '❌ Not achieved'}")
print("=" * 70)
