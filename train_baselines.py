# Trains the unimodal and multimodal baselines described in paper §4, using the cached features produced by extract_features.py. Saves per-fold F1 scores, per-class diagnostics, and timing information to CSV for later analysis.

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from loso_utils import SESSIONS, ROOT, load_cached_features, get_fold_split

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_NAMES = ["hap", "sad", "ang", "neu"]

MODEL_NAMES = [
    "Text only", "Audio only", "Video only",
    "Text + Audio", "Audio + Video", "Text + Video",
    "Early Fusion", "Late Fusion",
]


# Records the training and inference time for each model and fold, so that we can report the average per-fold time in the paper.
class ExperimentTimer:

    def __init__(self):
        self.rows = []

    def record(self, fold, model_name, stage, duration):
        self.rows.append({"fold": fold, "model": model_name, "stage": stage, "seconds": duration})

    def to_df(self):
        return pd.DataFrame(self.rows)


timer = ExperimentTimer()


def logreg():
    return LogisticRegression(
        max_iter=1000, C=1.0, multi_class="multinomial",
        solver="lbfgs", class_weight="balanced",
    )

# Fits the given classifier on the training set, times the training and inference, and returns the fitted classifier, the weighted F1 score on the test set, and the predicted labels. The timer records are stored in the global ExperimentTimer instance.
def score_instrumented(clf, X_tr, y_tr, X_te, y_te, fold_session, model_name):
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    timer.record(fold_session, model_name, "train", time.perf_counter() - t0)

    t1 = time.perf_counter()
    preds = clf.predict(X_te)
    inf_time = time.perf_counter() - t1
    timer.record(fold_session, model_name, "inference", inf_time)
    timer.record(fold_session, model_name, "per_utterance", inf_time / len(X_te))

    return clf, f1_score(y_te, preds, average="weighted"), preds

# Trains a binary classifier for each emotion (one-vs-rest) and records the per-class TP, TN, FP, FN, and F1 score in the given binary_results list. The classifier is trained on the given training set and evaluated on the test set. The label encoder is used to map emotion names to integer labels.
def train_binary_classifiers(model_name, X_tr, X_te, y_tr, y_te, le, fold_session, binary_results):
    for emo in EMOTION_NAMES:
        emo_idx = list(le.classes_).index(emo)
        yb_tr = (y_tr == emo_idx).astype(int)
        yb_te = (y_te == emo_idx).astype(int)

        clf_bin = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs")
        clf_bin.fit(X_tr, yb_tr)
        preds_bin = clf_bin.predict(X_te)

        tp = int(((preds_bin == 1) & (yb_te == 1)).sum())
        tn = int(((preds_bin == 0) & (yb_te == 0)).sum())
        fp = int(((preds_bin == 1) & (yb_te == 0)).sum())
        fn = int(((preds_bin == 0) & (yb_te == 1)).sum())
        f1 = f1_score(yb_te, preds_bin, zero_division=0)

        binary_results.append({
            "fold": fold_session, "model": model_name, "emotion": emo,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "f1": f1,
        })

# Entropy-based confidence measure for a given set of predicted probabilities. The static_weight parameter allows scaling the confidence by a fixed factor. This function is used in the late fusion baseline to weight the unimodal predictions by their confidence.
def refined_confidence(probs, static_weight):
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    sharpness = 1 / (entropy + 1e-6)
    return static_weight * sharpness


def run_fold(df_all, X_text, X_audio, X_video, y, le, fold_session,
             fold_results, binary_results, pooled_predictions):
    print(f"\n{'='*60}\nFOLD: test = Session {fold_session}\n{'='*60}")

    split = get_fold_split(df_all, X_text, X_audio, X_video, y, fold_session, save_scalers=True)
    Xt_tr, Xt_te = split["Xt_tr"], split["Xt_te"]
    Xa_tr, Xa_te = split["Xa_tr"], split["Xa_te"]
    Xv_tr, Xv_te = split["Xv_tr"], split["Xv_te"]
    y_tr, y_te = split["y_tr"], split["y_te"]

    Xta_tr = np.concatenate([Xt_tr, Xa_tr], axis=1); Xta_te = np.concatenate([Xt_te, Xa_te], axis=1)
    Xav_tr = np.concatenate([Xa_tr, Xv_tr], axis=1); Xav_te = np.concatenate([Xa_te, Xv_te], axis=1)
    Xtv_tr = np.concatenate([Xt_tr, Xv_tr], axis=1); Xtv_te = np.concatenate([Xt_te, Xv_te], axis=1)
    Xf_tr  = np.concatenate([Xt_tr, Xa_tr, Xv_tr], axis=1)
    Xf_te  = np.concatenate([Xt_te, Xa_te, Xv_te], axis=1)

    # Unimodal baselines (text, audio, video) and early fusion (concatenate all three, single classifier)
    clf_t, f1_t, preds_t = score_instrumented(logreg(), Xt_tr, y_tr, Xt_te, y_te, fold_session, "Text only")
    train_binary_classifiers("Text only", Xt_tr, Xt_te, y_tr, y_te, le, fold_session, binary_results)

    clf_a, f1_a, preds_a = score_instrumented(logreg(), Xa_tr, y_tr, Xa_te, y_te, fold_session, "Audio only")
    train_binary_classifiers("Audio only", Xa_tr, Xa_te, y_tr, y_te, le, fold_session, binary_results)

    clf_v, f1_v, preds_v = score_instrumented(logreg(), Xv_tr, y_tr, Xv_te, y_te, fold_session, "Video only")
    train_binary_classifiers("Video only", Xv_tr, Xv_te, y_tr, y_te, le, fold_session, binary_results)

    # Bimodal baselines (text+audio, audio+video, text+video)
    _, f1_ta, _ = score_instrumented(logreg(), Xta_tr, y_tr, Xta_te, y_te, fold_session, "Text + Audio")
    train_binary_classifiers("Text + Audio", Xta_tr, Xta_te, y_tr, y_te, le, fold_session, binary_results)

    _, f1_av, _ = score_instrumented(logreg(), Xav_tr, y_tr, Xav_te, y_te, fold_session, "Audio + Video")
    train_binary_classifiers("Audio + Video", Xav_tr, Xav_te, y_tr, y_te, le, fold_session, binary_results)

    _, f1_tv, _ = score_instrumented(logreg(), Xtv_tr, y_tr, Xtv_te, y_te, fold_session, "Text + Video")
    train_binary_classifiers("Text + Video", Xtv_tr, Xtv_te, y_tr, y_te, le, fold_session, binary_results)

    # Early fusion (concatenate all three modalities, single classifier)
    clf_f, f1_f, preds_f = score_instrumented(logreg(), Xf_tr, y_tr, Xf_te, y_te, fold_session, "Early Fusion")
    train_binary_classifiers("Early Fusion", Xf_tr, Xf_te, y_tr, y_te, le, fold_session, binary_results)

    # Late fusion (weighted average of unimodal predictions, weighted by entropy-based confidence)
    t_late = time.perf_counter()
    pt = clf_t.predict_proba(Xt_te)
    pa = clf_a.predict_proba(Xa_te)
    pv = clf_v.predict_proba(Xv_te)
    ct = refined_confidence(pt, 1 / 3)
    ca = refined_confidence(pa, 1 / 3)
    cv = refined_confidence(pv, 1 / 3)
    s = ct + ca + cv
    fused_probs = (ct / s)[:, None] * pt + (ca / s)[:, None] * pa + (cv / s)[:, None] * pv
    preds_late = np.argmax(fused_probs, axis=1)
    f1_late = f1_score(y_te, preds_late, average="weighted")
    timer.record(fold_session, "Late Fusion", "inference", time.perf_counter() - t_late)
    timer.record(fold_session, "Late Fusion", "train", 0.0)  # no separate training step
    train_binary_classifiers("Late Fusion", Xf_tr, Xf_te, y_tr, y_te, le, fold_session, binary_results)

    fold_results["Text only"].append(f1_t)
    fold_results["Audio only"].append(f1_a)
    fold_results["Video only"].append(f1_v)
    fold_results["Text + Audio"].append(f1_ta)
    fold_results["Audio + Video"].append(f1_av)
    fold_results["Text + Video"].append(f1_tv)
    fold_results["Early Fusion"].append(f1_f)
    fold_results["Late Fusion"].append(f1_late)

    pooled_predictions["Early Fusion"]["y_true"].extend(y_te)
    pooled_predictions["Early Fusion"]["y_pred"].extend(preds_f)
    pooled_predictions["Late Fusion"]["y_true"].extend(y_te)
    pooled_predictions["Late Fusion"]["y_pred"].extend(preds_late)

    print(f"  text={f1_t:.3f}  audio={f1_a:.3f}  video={f1_v:.3f}  "
          f"t+a={f1_ta:.3f}  a+v={f1_av:.3f}  t+v={f1_tv:.3f}  "
          f"early={f1_f:.3f}  late={f1_late:.3f}")


def main():
    df_all, X_text, X_audio, X_video, y, le = load_cached_features()

    fold_results = {name: [] for name in MODEL_NAMES}
    binary_results = []
    pooled_predictions = {
        name: {"y_true": [], "y_pred": []} for name in ["Early Fusion", "Late Fusion"]
    }

    for fold_session in SESSIONS:
        run_fold(df_all, X_text, X_audio, X_video, y, le, fold_session,
                 fold_results, binary_results, pooled_predictions)

    # Summary
    print(f"\n{'='*70}\n{'Model':<16} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'S5':>6} {'Mean':>7} {'±Std':>6}\n{'-'*70}")
    rows = []
    for model_name, scores in fold_results.items():
        arr = np.array(scores)
        mean, std = arr.mean(), arr.std()
        cv = std / mean if mean > 0 else 0.0
        print(f"{model_name:<16} " + " ".join(f"{s:.3f}" for s in arr) + f"  {mean:>6.3f}  ±{std:.3f}")
        rows.append({
            "Model": model_name,
            "S1": arr[0], "S2": arr[1], "S3": arr[2], "S4": arr[3], "S5": arr[4],
            "Mean F1": mean, "Std F1": std, "CV": cv,
        })

    results_df = pd.DataFrame(rows).round(4)
    results_df.to_csv(RESULTS_DIR / "baselines_results.csv", index=False)

    pd.DataFrame(binary_results).to_csv(RESULTS_DIR / "baselines_per_class.csv", index=False)
    timer.to_df().to_csv(RESULTS_DIR / "baselines_timing.csv", index=False)

    for model_name, data in pooled_predictions.items():
        print(f"\n{model_name} — pooled classification report (all 5 folds):")
        print(classification_report(data["y_true"], data["y_pred"], target_names=le.classes_))

    print(f"\nSaved: {RESULTS_DIR / 'baselines_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'baselines_per_class.csv'}")
    print(f"Saved: {RESULTS_DIR / 'baselines_timing.csv'}")


if __name__ == "__main__":
    main()