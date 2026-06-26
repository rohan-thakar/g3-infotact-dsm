# ============================================================
# MODEL 8 — Neural Network (MLP) — PyTorch
# Day 8 commit | Deep learning approach, dropout regularization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, f1_score,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
df['HDF_cond']      = ((df['Temp_diff'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
df['PWF_cond']      = ((df['Power_W'] < 3500) | (df['Power_W'] > 9000)).astype(int)
df['OSF_cond']      = (df['Wear_x_Torque'] > 11000).astype(int)
df['fault_risk']    = df['HDF_cond'] + df['PWF_cond'] + df['OSF_cond']

TARGET = 'Machine failure'
FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_cols = [c for c in df.columns if c not in [TARGET] + FAILURE_COLS]
X = df[feature_cols].values.astype(np.float32)
y = df[TARGET].values.astype(np.int64)

# ---- MLP ARCHITECTURE ----
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# ---- TRAINING FUNCTION ----
def train_model(X_train, y_train, X_val, y_val, n_epochs=50, lr=1e-3):
    scaler_nn = StandardScaler()
    X_train_sc = scaler_nn.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler_nn.transform(X_val).astype(np.float32)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

    X_tensor = torch.FloatTensor(X_train_sm).to(device)
    y_tensor = torch.LongTensor(y_train_sm).to(device)
    X_val_t  = torch.FloatTensor(X_val_sc).to(device)

    # Class weights for loss
    class_counts = np.bincount(y_train_sm)
    weights = 1.0 / class_counts
    class_weights = torch.FloatTensor(weights).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    model     = MLP(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
    return preds, probs, scaler_nn

# ---- 5-FOLD CV ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y), dtype=int)
oof_prob = np.zeros(len(y))
fold_f1s = []

print("Running 5-Fold CV for MLP...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    preds, probs, _ = train_model(X[train_idx], y[train_idx],
                                   X[val_idx],   y[val_idx])
    oof_pred[val_idx] = preds
    oof_prob[val_idx] = probs
    fold_f1 = f1_score(y[val_idx], preds, average='macro')
    fold_f1s.append(fold_f1)
    print(f"  Fold {fold+1}/5 → Macro F1: {fold_f1:.4f}")

print(f"\nMean CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

print("=" * 60)
print("MODEL 8: Neural Network (MLP)")
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
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

precision, recall, _ = precision_recall_curve(y, oof_prob)
ap = average_precision_score(y, oof_prob)
axes[0].plot(recall, precision, lw=2, color='#762a83', label=f'AP={ap:.3f}')
axes[0].fill_between(recall, precision, alpha=0.12, color='#762a83')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('PR Curve — MLP Neural Network')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y, oof_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['No Fail', 'Fail'], yticklabels=['No Fail', 'Fail'])
axes[1].set_title('Confusion Matrix — MLP')

plt.tight_layout()
plt.savefig('model8_mlp_neural_network.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model 8 complete.")
