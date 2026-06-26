# ============================================================
# MODEL 9 — Voting Ensemble (RF + XGBoost + LightGBM)
# Day 9 commit | Combines best models, soft voting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
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

TARGET = 'Machine failure'
FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_cols = [c for c in df.columns if c not in [TARGET] + FAILURE_COLS]
X = df[feature_cols]
y = df[TARGET]

# ---- BASE MODELS ----
neg, pos = (y == 0).sum(), (y == 1).sum()
spw = neg / pos

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=12, max_features='sqrt',
    class_weight='balanced', n_jobs=-1, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, use_label_encoder=False,
    eval_metric='logloss', random_state=42, n_jobs=-1)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=7, num_leaves=40,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)

# ---- SOFT VOTING ENSEMBLE ----
ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgb', lgb_model)],
    voting='soft',
    weights=[1, 1.5, 1.5]   # slightly higher weight to gradient boosters
)

# ---- SMOTE INSIDE CV ----
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

oof_pred = np.zeros(len(y), dtype=int)
oof_prob = np.zeros(len(y))
fold_f1s = []

print("Running 5-Fold CV for Voting Ensemble (RF + XGB + LGB)...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    ensemble.fit(X_tr_sm, y_tr_sm)
    oof_pred[val_idx] = ensemble.predict(X_val)
    oof_prob[val_idx] = ensemble.predict_proba(X_val)[:, 1]

    fold_f1 = f1_score(y_val, oof_pred[val_idx], average='macro')
    fold_f1s.append(fold_f1)
    print(f"  Fold {fold+1}/5 → Macro F1: {fold_f1:.4f}")

print(f"\nMean CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

print("=" * 60)
print("MODEL 9: Voting Ensemble (RF + XGBoost + LightGBM)")
print("=" * 60)
print(classification_report(y, oof_pred, target_names=['No Failure', 'Failure']))
macro_f1 = f1_score(y, oof_pred, average='macro')
roc_auc  = roc_auc_score(y, oof_prob)
print(f"Macro F1 Score : {macro_f1:.4f}  (target ≥ 0.85)")
print(f"ROC-AUC Score  : {roc_auc:.4f}")

best_thresh, best_f1 = 0.5, 0
for thresh in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y, (oof_prob >= thresh).astype(int), average='macro')
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh
print(f"\nBest threshold: {best_thresh:.2f} → Macro F1: {best_f1:.4f}")

# ---- PLOTS ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

precision, recall, _ = precision_recall_curve(y, oof_prob)
ap = average_precision_score(y, oof_prob)
axes[0].plot(recall, precision, lw=2.5, color='#e08214', label=f'AP={ap:.3f}')
axes[0].fill_between(recall, precision, alpha=0.15, color='#e08214')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('PR Curve — Voting Ensemble')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y, oof_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['No Fail', 'Fail'], yticklabels=['No Fail', 'Fail'])
axes[1].set_title('Confusion Matrix — Voting Ensemble')

plt.tight_layout()
plt.savefig('model9_voting_ensemble.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model 9 complete.")
