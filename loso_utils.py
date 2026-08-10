# Shared 5-fold LOSO-CV

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

SESSIONS = [1, 2, 3, 4, 5]

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "cached_features"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Loads the cached features and labels from disk, returning the utterance-level index (df_all), the three modality matrices, the integer labels, and the fitted label encoder.
def load_cached_features():
    df_all = pd.read_csv(FEATURE_DIR / "df_all.csv")

    X_text = np.vstack([np.load(p) for p in df_all["text_feature_path"]])
    X_audio = np.vstack([np.load(p) for p in df_all["audio_feature_path"]])
    X_video = np.vstack([np.load(p) for p in df_all["visual_feature_path"]])

    le = LabelEncoder()
    y = le.fit_transform(df_all["emotion"])

    return df_all, X_text, X_audio, X_video, y, le

# Normalizes the training and test sets per speaker, using the training set to fit the mean/std for each speaker. If skip_zeros=True, any rows that are all zeros (e.g., missing visual features) are skipped when fitting the per-speaker stats.
def _speaker_norm(X_tr, X_te, spk_tr, spk_te, skip_zeros=False):
    stats = {}
    for spk in np.unique(spk_tr):
        mask = spk_tr == spk
        rows = X_tr[mask]
        if skip_zeros:
            valid = ~np.all(rows == 0, axis=1)
            rows = rows[valid]
        if len(rows) == 0:
            continue
        stats[spk] = {"mean": rows.mean(0), "std": rows.std(0)}

    X_tr_n = X_tr.copy()
    for i, spk in enumerate(spk_tr):
        if spk in stats and (not skip_zeros or not np.all(X_tr[i] == 0)):
            X_tr_n[i] = (X_tr[i] - stats[spk]["mean"]) / (stats[spk]["std"] + 1e-6)

    X_te_n = X_te.copy()
    for i, spk in enumerate(spk_te):
        if spk in stats and (not skip_zeros or not np.all(X_te[i] == 0)):
            X_te_n[i] = (X_te[i] - stats[spk]["mean"]) / (stats[spk]["std"] + 1e-6)

    return X_tr_n, X_te_n, stats

# Builds the train/test split for a given fold/session, normalizing the features and optionally saving the fitted scalers to disk. Returns a dictionary containing the train/test splits for each modality, the labels, and the train/test indices.
def get_fold_split(df_all, X_text, X_audio, X_video, y, fold_session, save_scalers=False):
    test_mask = df_all["utterance_id"].str.startswith(f"Ses0{fold_session}")
    train_mask = ~test_mask
    tr_idx = df_all.index[train_mask].tolist()
    te_idx = df_all.index[test_mask].tolist()

    y_tr, y_te = y[tr_idx], y[te_idx]

    Xt_tr, Xt_te = X_text[tr_idx], X_text[te_idx]
    Xa_tr, Xa_te = X_audio[tr_idx], X_audio[te_idx]
    Xv_tr, Xv_te = X_video[tr_idx], X_video[te_idx]

    sc_t = StandardScaler(); Xt_tr = sc_t.fit_transform(Xt_tr); Xt_te = sc_t.transform(Xt_te)
    sc_a = StandardScaler(); Xa_tr = sc_a.fit_transform(Xa_tr); Xa_te = sc_a.transform(Xa_te)
    sc_v = StandardScaler(); Xv_tr = sc_v.fit_transform(Xv_tr); Xv_te = sc_v.transform(Xv_te)

    spk_tr = df_all.loc[tr_idx, "speaker"].values
    spk_te = df_all.loc[te_idx, "speaker"].values

    Xa_tr, Xa_te, speaker_stats_audio = _speaker_norm(Xa_tr, Xa_te, spk_tr, spk_te, skip_zeros=False)
    Xv_tr, Xv_te, speaker_stats_video = _speaker_norm(Xv_tr, Xv_te, spk_tr, spk_te, skip_zeros=True)

    if save_scalers:
        joblib.dump(sc_t, CKPT_DIR / f"scaler_text_fold{fold_session}.joblib")
        joblib.dump(sc_a, CKPT_DIR / f"scaler_audio_fold{fold_session}.joblib")
        joblib.dump(sc_v, CKPT_DIR / f"scaler_video_fold{fold_session}.joblib")
        joblib.dump(speaker_stats_audio, CKPT_DIR / f"speaker_stats_audio_fold{fold_session}.joblib")
        joblib.dump(speaker_stats_video, CKPT_DIR / f"speaker_stats_video_fold{fold_session}.joblib")

    return {
        "Xt_tr": Xt_tr, "Xt_te": Xt_te,
        "Xa_tr": Xa_tr, "Xa_te": Xa_te,
        "Xv_tr": Xv_tr, "Xv_te": Xv_te,
        "y_tr": y_tr, "y_te": y_te,
        "tr_idx": tr_idx, "te_idx": te_idx,
    }