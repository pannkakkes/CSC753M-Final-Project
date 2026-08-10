# Trains and evaluates the proposed CrossAttentionFusion model on the IEMOCAP dataset using a 5-fold LOSO-CV setup. The script first performs a hyperparameter search on an internal validation split, then trains and evaluates the model for each fold and anchor type (text, audio, video, avg). Results are saved to CSV files for later analysis.

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from fusion_model import CrossAttentionFusion, IEMOCAPDataset
from loso_utils import SESSIONS, ROOT, CKPT_DIR, load_cached_features, get_fold_split

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_NAMES = ["hap", "sad", "ang", "neu"]
ANCHOR_NAMES = ["text", "audio", "video", "avg"]
MODEL_NAMES = [f"Attn ({a}-anchor)" for a in ANCHOR_NAMES]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentTimer:
    def __init__(self):
        self.rows = []

    def record(self, fold, model_name, stage, duration):
        self.rows.append({"fold": fold, "model": model_name, "stage": stage, "seconds": duration})

    def to_df(self):
        return pd.DataFrame(self.rows)


timer = ExperimentTimer()


# Hyperparameter search space for the CrossAttentionFusion model. The search space includes learning rate, batch size, dropout rate, number of attention heads, and embedding dimension. The search is performed using random sampling within the specified ranges.

HP_SEARCH_SPACE = {
    "lr": (1e-5, 1e-3),          # sampled log-uniform
    "batch_size": [32, 64],
    "dropout": (0.1, 0.5),       # sampled uniform
    "num_heads": [4, 8, 16],
    "embed_dim": [64, 128, 256],
}
N_TRIALS = 20


def sample_hparams(rng):
    log_lr = rng.uniform(np.log10(HP_SEARCH_SPACE["lr"][0]), np.log10(HP_SEARCH_SPACE["lr"][1]))
    return {
        "lr": float(10 ** log_lr),
        "batch_size": int(rng.choice(HP_SEARCH_SPACE["batch_size"])),
        "dropout": float(rng.uniform(*HP_SEARCH_SPACE["dropout"])),
        "num_heads": int(rng.choice(HP_SEARCH_SPACE["num_heads"])),
        "embed_dim": int(rng.choice(HP_SEARCH_SPACE["embed_dim"])),
    }


def train_and_validate(hparams, Xt_tr, Xa_tr, Xv_tr, y_tr, Xt_val, Xa_val, Xv_val, y_val,
                        t_in, a_in, v_in, anchor="avg", epochs=40, max_patience=8):
    class_counts = np.bincount(y_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE))

    ds = IEMOCAPDataset(Xt_tr, Xa_tr, Xv_tr, y_tr)
    loader = DataLoader(ds, batch_size=hparams["batch_size"], shuffle=True)

    m = CrossAttentionFusion(
        t_in, a_in, v_in, embed_dim=hparams["embed_dim"], num_heads=hparams["num_heads"],
        anchor=anchor, dropout=hparams["dropout"],
    ).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=hparams["lr"], weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_loss, patience_ctr = float("inf"), 0
    for ep in range(epochs):
        m.train()
        total = 0
        for t, a, v, yb in loader:
            t, a, v, yb = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(m(t, a, v), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        avg_loss = total / len(loader)
        sch.step(avg_loss)
        if avg_loss < best_loss:
            best_loss, patience_ctr = avg_loss, 0
        else:
            patience_ctr += 1
        if patience_ctr >= max_patience:
            break

    m.eval()
    with torch.no_grad():
        t = torch.tensor(Xt_val, dtype=torch.float32).to(DEVICE)
        a = torch.tensor(Xa_val, dtype=torch.float32).to(DEVICE)
        v = torch.tensor(Xv_val, dtype=torch.float32).to(DEVICE)
        preds = m(t, a, v).argmax(dim=1).cpu().numpy()
    return f1_score(y_val, preds, average="weighted")

# Carves out an internal validation split from the training set for hyperparameter search. The last session in the training set is held out as the internal validation set, while the remaining sessions are used for training. This ensures that the hyperparameter search is performed on data that is not seen during model training, providing a more realistic estimate of model performance on unseen data.
def run_hyperparameter_search(df_all, X_text, X_audio, X_video, y):
    split = get_fold_split(df_all, X_text, X_audio, X_video, y, fold_session=5, save_scalers=False)
    tr_idx = split["tr_idx"]

    session_for_tr = df_all.loc[tr_idx, "utterance_id"].str.extract(r'(Ses\d+)')[0].values
    train_sessions_available = sorted(np.unique(session_for_tr))
    internal_val_session = train_sessions_available[-1]  # last training session held out
    internal_train_mask = session_for_tr != internal_val_session
    internal_val_mask = ~internal_train_mask

    Xt_hp_tr, Xt_hp_val = split["Xt_tr"][internal_train_mask], split["Xt_tr"][internal_val_mask]
    Xa_hp_tr, Xa_hp_val = split["Xa_tr"][internal_train_mask], split["Xa_tr"][internal_val_mask]
    Xv_hp_tr, Xv_hp_val = split["Xv_tr"][internal_train_mask], split["Xv_tr"][internal_val_mask]
    y_hp_tr, y_hp_val = split["y_tr"][internal_train_mask], split["y_tr"][internal_val_mask]

    print(f"HP search: internal-train={len(y_hp_tr)} samples, "
          f"internal-val=session {internal_val_session} ({len(y_hp_val)} samples)")

    rng = np.random.default_rng(42)
    t_in, a_in, v_in = X_text.shape[1], X_audio.shape[1], X_video.shape[1]

    search_log = []
    print(f"\n=== Random search: {N_TRIALS} trials (avg-anchor, internal val only) ===")
    for trial in range(N_TRIALS):
        hparams = sample_hparams(rng)
        val_f1 = train_and_validate(
            hparams, Xt_hp_tr, Xa_hp_tr, Xv_hp_tr, y_hp_tr, Xt_hp_val, Xa_hp_val, Xv_hp_val, y_hp_val,
            t_in, a_in, v_in, anchor="avg",
        )
        hparams["val_f1"] = val_f1
        search_log.append(hparams)
        print(f"  trial {trial:2d} | f1={val_f1:.4f} | {hparams}")

    search_df = pd.DataFrame(search_log).sort_values("val_f1", ascending=False)
    search_df.to_csv(RESULTS_DIR / "hyperparameter_search_log.csv", index=False)

    best_hp = search_df.iloc[0].to_dict()
    print(f"\nBest hyperparameters (val F1={best_hp['val_f1']:.4f}):")
    print({k: v for k, v in best_hp.items() if k != "val_f1"})

    with open(CKPT_DIR / "cross_attn_config.json", "w") as f:
        json.dump({
            "t_in": t_in, "a_in": a_in, "v_in": v_in,
            "embed_dim": int(best_hp["embed_dim"]), "num_heads": int(best_hp["num_heads"]),
            "dropout": float(best_hp["dropout"]),
        }, f)
    print(f"Saved config to {CKPT_DIR / 'cross_attn_config.json'}")

    return best_hp


# Main training loop for the CrossAttentionFusion model. For each fold in the LOSO-CV setup, the function trains and evaluates the model for each anchor type (text, audio, video, avg). The best hyperparameters from the internal validation search are used for training. Results, including F1 scores and timing information, are recorded and saved to CSV files for later analysis.

# Post-hoc binary classifiers are trained for each emotion class using the predictions from the multi-class model. This allows for a more detailed analysis of the model's performance on individual emotion classes, providing insights into which emotions are more challenging to classify and where improvements can be made.
def train_binary_classifiers_torch(model_name, preds, y_te, fold_session, binary_results):
    for emo_idx, emo in enumerate(EMOTION_NAMES):
        yb_true = (y_te == emo_idx).astype(int)
        yb_pred = (preds == emo_idx).astype(int)
        tp = int(((yb_pred == 1) & (yb_true == 1)).sum())
        tn = int(((yb_pred == 0) & (yb_true == 0)).sum())
        fp = int(((yb_pred == 1) & (yb_true == 0)).sum())
        fn = int(((yb_pred == 0) & (yb_true == 1)).sum())
        f1 = f1_score(yb_true, yb_pred, zero_division=0)
        binary_results.append({
            "fold": fold_session, "model": model_name, "emotion": emo,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "f1": f1,
        })


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for t, a, v, y in loader:
            t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
            preds = torch.argmax(model(t, a, v), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# Trains the CrossAttentionFusion model for a single fold and anchor type. The function takes the training data loader, input dimensions, loss criterion, and best hyperparameters as input. It trains the model for a specified number of epochs, using early stopping based on validation loss. The trained model is returned for evaluation on the test set.
def train_fold_variant(anchor, train_loader, t_in, a_in, v_in, criterion, best_hp,
                        epochs=150, max_patience=15):
    m = CrossAttentionFusion(
        t_in, a_in, v_in, anchor=anchor,
        embed_dim=int(best_hp["embed_dim"]), num_heads=int(best_hp["num_heads"]),
        dropout=float(best_hp["dropout"]),
    ).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=float(best_hp["lr"]), weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    best_loss, patience_ctr = float('inf'), 0
    for ep in range(epochs):
        m.train()
        total = 0
        for t, a, v, yb in train_loader:
            t, a, v, yb = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(m(t, a, v), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        avg_loss = total / len(train_loader)
        sched.step(avg_loss)
        if avg_loss < best_loss:
            best_loss, patience_ctr = avg_loss, 0
        else:
            patience_ctr += 1
        if patience_ctr >= max_patience:
            break
    return m


def run_fold(df_all, X_text, X_audio, X_video, y, le, fold_session, best_hp,
             fold_results, binary_results, pooled_predictions):
    print(f"\n{'='*60}\nFOLD: test = Session {fold_session}\n{'='*60}")

    split = get_fold_split(df_all, X_text, X_audio, X_video, y, fold_session, save_scalers=True)
    Xt_tr, Xt_te = split["Xt_tr"], split["Xt_te"]
    Xa_tr, Xa_te = split["Xa_tr"], split["Xa_te"]
    Xv_tr, Xv_te = split["Xv_tr"], split["Xv_te"]
    y_tr, y_te = split["y_tr"], split["y_te"]

    t_in, a_in, v_in = X_text.shape[1], X_audio.shape[1], X_video.shape[1]

    class_counts = np.bincount(y_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE))

    train_ds = IEMOCAPDataset(Xt_tr, Xa_tr, Xv_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=int(best_hp["batch_size"]), shuffle=True)
    test_ds = IEMOCAPDataset(Xt_te, Xa_te, Xv_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    for anchor in ANCHOR_NAMES:
        model_name = f"Attn ({anchor}-anchor)"

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = train_fold_variant(anchor, train_loader, t_in, a_in, v_in, criterion, best_hp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.record(fold_session, model_name, "train", time.perf_counter() - t0)

        torch.save(m.state_dict(), CKPT_DIR / f"cross_attn_{anchor}anchor_fold{fold_session}.pt")

        t1 = time.perf_counter()
        y_true_f, preds = evaluate_model(m, test_loader, DEVICE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf_time = time.perf_counter() - t1
        timer.record(fold_session, model_name, "inference", inf_time)
        timer.record(fold_session, model_name, "per_utterance", inf_time / len(y_te))

        f1 = f1_score(y_true_f, preds, average="weighted")
        fold_results[model_name].append(f1)
        pooled_predictions[model_name]["y_true"].extend(y_true_f)
        pooled_predictions[model_name]["y_pred"].extend(preds)
        train_binary_classifiers_torch(model_name, preds, y_te, fold_session, binary_results)

        print(f"  [{anchor}-anchor] fold {fold_session} F1={f1:.4f}")

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import gc
    gc.collect()


def main():
    df_all, X_text, X_audio, X_video, y, le = load_cached_features()

    best_hp = run_hyperparameter_search(df_all, X_text, X_audio, X_video, y)

    fold_results = {name: [] for name in MODEL_NAMES}
    binary_results = []
    pooled_predictions = {name: {"y_true": [], "y_pred": []} for name in MODEL_NAMES}

    for fold_session in SESSIONS:
        run_fold(df_all, X_text, X_audio, X_video, y, le, fold_session, best_hp,
                 fold_results, binary_results, pooled_predictions)

    # Parameter count for the CrossAttentionFusion model with the best hyperparameters. This provides insight into the model's complexity and potential for overfitting. The number of parameters is printed to the console for reference.
    sample_model = CrossAttentionFusion(
        X_text.shape[1], X_audio.shape[1], X_video.shape[1],
        embed_dim=int(best_hp["embed_dim"]), num_heads=int(best_hp["num_heads"]),
        dropout=float(best_hp["dropout"]), anchor="avg",
    )
    n_params = sum(p.numel() for p in sample_model.parameters())
    print(f"\nCrossAttentionFusion parameters: {n_params:,}")

    # Summary
    print(f"\n{'='*70}\n{'Model':<20} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'S5':>6} {'Mean':>7} {'±Std':>6}\n{'-'*70}")
    rows = []
    for model_name, scores in fold_results.items():
        arr = np.array(scores)
        mean, std = arr.mean(), arr.std()
        cv = std / mean if mean > 0 else 0.0
        print(f"{model_name:<20} " + " ".join(f"{s:.3f}" for s in arr) + f"  {mean:>6.3f}  ±{std:.3f}")
        rows.append({
            "Model": model_name,
            "S1": arr[0], "S2": arr[1], "S3": arr[2], "S4": arr[3], "S5": arr[4],
            "Mean F1": mean, "Std F1": std, "CV": cv, "Params": n_params,
        })

    pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "cross_attention_results.csv", index=False)
    pd.DataFrame(binary_results).to_csv(RESULTS_DIR / "cross_attention_per_class.csv", index=False)
    timer.to_df().to_csv(RESULTS_DIR / "cross_attention_timing.csv", index=False)

    for model_name, data in pooled_predictions.items():
        print(f"\n{model_name} — pooled classification report (all 5 folds):")
        print(classification_report(data["y_true"], data["y_pred"], target_names=le.classes_))

    print(f"\nSaved: {RESULTS_DIR / 'cross_attention_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cross_attention_per_class.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cross_attention_timing.csv'}")
    print(f"Saved per-fold checkpoints to: {CKPT_DIR}/")


if __name__ == "__main__":
    main()