"""
==============================================================================
Phase 3: Anomaly Detection & Risk Scoring for Mule Account Detection
==============================================================================
This module adds an unsupervised anomaly detection layer to catch
novel mule patterns not seen in historical labeled data.

Methods Implemented:
  1. Isolation Forest    — samples partitioning for outlier scoring
  2. ECOD               — Empirical Cumulative Outlier Detection (fast, no params)
  3. Autoencoder        — Neural network reconstruction error as anomaly signal
  4. Risk Score Fusion  — Combines supervised model score + anomaly score

Why Anomaly Detection?
  - The labeled dataset may capture KNOWN mule patterns.
  - Fraudsters evolve. New mule typologies produce no labels initially.
  - Combining supervised + unsupervised builds a defense-in-depth system.
==============================================================================
"""
import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import joblib

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch not available. Autoencoder anomaly detection skipped.")

warnings.filterwarnings('ignore')

from shared_config import MODEL_REPORTS_DIR, MODELS_DIR, load_engineered_data

# ── CONFIG ───────────────────────────────────────────────────────────────────
REPORTS_DIR  = MODEL_REPORTS_DIR
TARGET_COL   = "F3924"
RANDOM_STATE = 42
ISOFOREST_CONTAMINATION = 0.01  # Expected % of anomalies (~1%)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading Engineered Dataset")
print("=" * 70)
df = load_engineered_data()
feature_cols = [c for c in df.columns if c != TARGET_COL]
X_raw = df[feature_cols].values.astype(np.float32)
y     = df[TARGET_COL].values.astype(int)

# Robust scale features (Isolation Forest & Autoencoder benefit from scaling)
scaler = RobustScaler()
X = scaler.fit_transform(X_raw)
joblib.dump(scaler, f"{MODELS_DIR}/robust_scaler_anomaly.pkl")
print(f"  Shape: {X.shape}  |  Fraud rate: {y.mean()*100:.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("METHOD 1: Isolation Forest")
print("=" * 70)
iso_forest = IsolationForest(
    n_estimators    = 500,
    contamination   = ISOFOREST_CONTAMINATION,
    max_samples     = 'auto',
    random_state    = RANDOM_STATE,
    n_jobs          = -1,
)
iso_forest.fit(X[y == 0])  # Train ONLY on legitimate accounts (semi-supervised approach)
iso_scores_raw = iso_forest.decision_function(X)       # Higher = more normal
# Normalize to [0, 1] with inversion (higher score = more suspicious)
iso_anomaly_score = 1 - (iso_scores_raw - iso_scores_raw.min()) / \
                        (iso_scores_raw.max() - iso_scores_raw.min())

iso_auc = roc_auc_score(y, iso_anomaly_score)
iso_ap  = average_precision_score(y, iso_anomaly_score)
print(f"  Isolation Forest — ROC-AUC: {iso_auc:.4f}  |  PR-AUC: {iso_ap:.4f}")
joblib.dump(iso_forest, f"{MODELS_DIR}/isolation_forest.pkl")
print(f"  [Saved] {MODELS_DIR}/isolation_forest.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: ECOD (Empirical Cumulative Outlier Detection)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("METHOD 2: ECOD — Empirical Cumulative Outlier Detection")
print("=" * 70)
try:
    from pyod.models.ecod import ECOD
    ecod = ECOD(contamination=ISOFOREST_CONTAMINATION, n_jobs=-1)
    ecod.fit(X[y == 0])
    ecod_scores = ecod.decision_function(X)
    ecod_scores_norm = (ecod_scores - ecod_scores.min()) / (ecod_scores.max() - ecod_scores.min())
    ecod_auc = roc_auc_score(y, ecod_scores_norm)
    ecod_ap  = average_precision_score(y, ecod_scores_norm)
    print(f"  ECOD — ROC-AUC: {ecod_auc:.4f}  |  PR-AUC: {ecod_ap:.4f}")
    joblib.dump(ecod, f"{MODELS_DIR}/ecod_model.pkl")
    print(f"  [Saved] {MODELS_DIR}/ecod_model.pkl")
except ImportError:
    print("  [SKIP] pyod not installed (pip install pyod). Skipping ECOD.")
    ecod_scores_norm = np.zeros(len(y))
    ecod_auc, ecod_ap = 0.5, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: Autoencoder (Neural Network Reconstruction Error)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("METHOD 3: Autoencoder — Reconstruction Error as Anomaly Score")
print("=" * 70)

autoencoder_recon_error = None

if TORCH_AVAILABLE:
    n_features = X.shape[1]

    class TransactionAutoencoder(nn.Module):
        """Bottleneck autoencoder. High reconstruction error → anomalous account."""
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 32),   # Bottleneck: compressed latent representation
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed

    # Train ONLY on legitimate accounts (model learns "normal" behavior)
    X_legit = torch.tensor(X[y == 0], dtype=torch.float32)
    X_all   = torch.tensor(X, dtype=torch.float32)

    EPOCHS     = 30
    BATCH_SIZE = 1024
    LR         = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on device: {device}")

    autoencoder = TransactionAutoencoder(n_features).to(device)
    optimizer   = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)
    criterion   = nn.MSELoss()

    dataset    = torch.utils.data.TensorDataset(X_legit)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    autoencoder.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            recon = autoencoder(batch_x)
            loss  = criterion(recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        avg_loss = epoch_loss / len(X_legit)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}]  Reconstruction Loss: {avg_loss:.6f}")

    # Compute per-sample reconstruction error on ALL samples
    autoencoder.eval()
    with torch.no_grad():
        recon_all = autoencoder(X_all.to(device))
        recon_err = torch.mean((X_all.to(device) - recon_all) ** 2, dim=1).cpu().numpy()

    autoencoder_recon_error = recon_err
    ae_score_norm = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min())
    ae_auc = roc_auc_score(y, ae_score_norm)
    ae_ap  = average_precision_score(y, ae_score_norm)
    print(f"  Autoencoder — ROC-AUC: {ae_auc:.4f}  |  PR-AUC: {ae_ap:.4f}")
    torch.save(autoencoder.state_dict(), f"{MODELS_DIR}/autoencoder_state.pt")
    print(f"  [Saved] {MODELS_DIR}/autoencoder_state.pt")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, color='#1E88E5', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Reconstruction Loss')
    ax.set_title('Autoencoder Training Loss on Legitimate Accounts', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/autoencoder_training_loss.png", dpi=150)
    plt.close()
else:
    ae_score_norm = np.zeros(len(y))
    ae_auc, ae_ap = 0.5, 0.0
    print("  [SKIP] PyTorch not available.")


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORE FUSION: Supervised + Anomaly Score Blend
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RISK SCORE FUSION: Combining Supervised + Anomaly Scores")
print("=" * 70)

# Load LGBM supervised probabilities (from Phase 2)
try:
    lgbm_model = joblib.load(f"{MODELS_DIR}/lgbm_final.pkl")
    supervised_prob = lgbm_model.predict_proba(
        load_engineered_data().drop(columns=[TARGET_COL]).values.astype(np.float32)
    )[:, 1]
except Exception as e:
    print(f"  [WARN] Could not load LightGBM model: {e}. Using Isolation Forest score as primary.")
    supervised_prob = iso_anomaly_score

# Weighted Combination:
# - Primary weight goes to supervised model (proven labels)
# - Secondary weight goes to anomaly signal (novelty detection)
W_SUPERVISED = 0.70
W_ANOMALY    = 0.30

# Best anomaly score = Isolation Forest (most stable without dependencies)
final_risk_score = W_SUPERVISED * supervised_prob + W_ANOMALY * iso_anomaly_score

# Metrics for fused score
fused_auc = roc_auc_score(y, final_risk_score)
fused_ap  = average_precision_score(y, final_risk_score)
print(f"  Fused Risk Score — ROC-AUC: {fused_auc:.4f}  |  PR-AUC: {fused_ap:.4f}")

# Save risk scores to dataset
df_out = df[[TARGET_COL]].copy()
df_out['supervised_prob']    = supervised_prob
df_out['iso_anomaly_score']  = iso_anomaly_score
df_out['final_risk_score']   = final_risk_score
df_out.to_csv(f"{REPORTS_DIR}/risk_scores_all_accounts.csv", index=False)
print(f"  [Saved] {REPORTS_DIR}/risk_scores_all_accounts.csv")

# ── Score Distribution Comparison ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (scores, name, color) in zip(axes, [
    (supervised_prob,   'Supervised (LightGBM)',   '#E53935'),
    (iso_anomaly_score, 'Isolation Forest Score',  '#1E88E5'),
    (final_risk_score,  'Fused Risk Score (Final)','#43A047'),
]):
    ax.hist(scores[y == 0], bins=80, alpha=0.6, color='#4CAF50', label='Legitimate', density=True)
    ax.hist(scores[y == 1], bins=80, alpha=0.7, color='#F44336', label='Mule/Suspect', density=True)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Risk Score Distributions — Legitimate vs Mule', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/risk_score_distributions.png", dpi=150)
plt.close()
print(f"  [Saved] {REPORTS_DIR}/risk_score_distributions.png")

print("\n" + "=" * 70)
print("ANOMALY DETECTION & RISK SCORING COMPLETE")
print("=" * 70)
