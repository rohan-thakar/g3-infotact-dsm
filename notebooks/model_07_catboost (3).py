# ============================================================
# MODEL 7 — CatBoost Classifier
# Day 7 commit | Handles categorical natively, robust to noise
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
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

# NOTE: CatBoost handles categorical directly — keep Type as is
# But still encode for SMOTE compatibility
le = LabelEncoder()
df['Type_enc'] = le.fit_transform(df['Type'])
df.drop(columns=['Type'], inplace=True)

df['Temp_diff']     = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power_W']       = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
df['Wear_x_Torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['Stress_proxy']  = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-6)
df['Wear_per_RPM']  = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1e-6)
df['Torque_dev']    = np.abs(df['Torque [Nm]'] - df['Torque [Nm]'].mean())
df['HDF_cond']      = ((df['Temp_diff'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
df['PWF_cond']      = ((df['Power_W'] < 3500) | (df['Power_W'] > 9000)).astype(int)
df['OSF_cond']      = (df['Wear_x_Torque'] > 11000).astype(int)
df['fault_risk']    = df['HDF_cond'] + df['PWF_cond'] + df['OSF_cond']

TARGET = 'Machine failure'
FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_cols = [c for c in df.columns if c not in [TARGET] + FAILURE_COLS]
X = df[feature_cols]
y = df[TARGET]

# ---- SMOTE INSIDE CV ----
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

oof_pred = np.zeros(len(y))
oof_prob = np.zeros(len(y))
fold_f1s = []

print("Running 5-Fold CV with SMOTE inside each fold...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        auto_class_weights='Balanced',
        eval_metric='F1',
        random_seed=42,
        verbose=False
    )
    model.fit(X_tr_sm, y_tr_sm,
              eval_set=(X_val, y_val),
              early_stopping_rounds=50,
              verbose=False)

    oof_pred[val_idx] = model.predict(X_val)
    oof_prob[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_f1 = f1_score(y_val, oof_pred[val_idx], average='macro')
    fold_f1s.append(fold_f1)
    print(f"  Fold {fold+1}/5 → Macro F1: {fold_f1:.4f}")

print(f"\nMean CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

print("=" * 60)
print("MODEL 7: CatBoost")
print("=" * 60)
print(classification_report(y, oof_pred, target_names=['No Failure', 'Failure']))
macro_f1 = f1_score(y, oof_pred, average='macro')
roc_auc  = roc_auc_score(y, oof_prob)
print(f"Macro F1 Score : {macro_f1:.4f}  (target ≥ 0.85)")
print(f"ROC-AUC Score  : {roc_auc:.4f}")

# ---- THRESHOLD TUNING ----
best_thresh, best_f1 = 0.5, 0
for thresh in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y, (oof_prob >= thresh).astype(int), average='macro')
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh
print(f"\nBest threshold: {best_thresh:.2f} → Macro F1: {best_f1:.4f}")

# ---- PLOTS ----
# Final model for feature importance
X_tr_sm, y_tr_sm = smote.fit_resample(X, y)
final_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=7,
                                  auto_class_weights='Balanced', random_seed=42, verbose=False)
final_model.fit(X_tr_sm, y_tr_sm)
importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

importances.head(12).plot(kind='barh', ax=axes[0], color='#d73027')
axes[0].set_title('Top 12 Features — CatBoost')
axes[0].invert_yaxis()

precision, recall, _ = precision_recall_curve(y, oof_prob)
ap = average_precision_score(y, oof_prob)
axes[1].plot(recall, precision, lw=2, color='#d73027', label=f'AP={ap:.3f}')
axes[1].fill_between(recall, precision, alpha=0.12, color='#d73027')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('PR Curve — CatBoost'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y, oof_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[2],
            xticklabels=['No Fail', 'Fail'], yticklabels=['No Fail', 'Fail'])
axes[2].set_title('Confusion Matrix — CatBoost')

plt.tight_layout()
plt.savefig('model7_catboost.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model 7 complete.")
