import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from fusion_model import CrossAttentionFusion
from loso_utils import ROOT, CKPT_DIR, load_cached_features, get_fold_split

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TSNE_FOLD = 5 

EMOTION_COLORS = {
    "ang": "#E63946",
    "hap": "#F4A261",
    "neu": "#457B9D",
    "sad": "#6A4C93",
}

# Merges the three training scripts' per-model results CSVs into a single {model_name: [S1..S5 weighted F1]} dict
def load_pooled_predictions(class_names):
    frames = []
    for fname in ["cross_attention_pooled_predictions.csv", "gcn_baselines_pooled_predictions.csv"]:
        path = RESULTS_DIR / fname
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"WARNING: {fname} not found — run its training script first.")
    if not frames:
        raise FileNotFoundError("No pooled-prediction CSVs found.")
    df = pd.concat(frames, ignore_index=True)

    pooled = {}
    for model_name, g in df.groupby("model"):
        pooled[model_name] = {"y_true": g["y_true"].tolist(), "y_pred": g["y_pred"].tolist()}
    return pooled


def plot_aggregate_confusion_matrices(pooled, class_names, models=("Attn (avg-anchor)", "MMGCN (avt)")):
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), dpi=150)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        if model_name not in pooled:
            print(f"WARNING: no pooled predictions for '{model_name}' — skipping in Figure 2.")
            continue
        data = pooled[model_name]
        print(f"\n{model_name} — pooled classification report (all 5 folds):")
        print(classification_report(data["y_true"], data["y_pred"], target_names=class_names))

        cm = confusion_matrix(data["y_true"], data["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
        ax.set_title(f"{model_name} — Aggregate Confusion Matrix (5-fold LOSO)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'figure2_confusion_matrices.png'}")

    # Also save each model's own confusion matrix individually
    for model_name, data in pooled.items():
        cm = confusion_matrix(data["y_true"], data["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
        disp.plot(cmap='Blues', ax=ax2, values_format='d')
        ax2.set_title(f"{model_name} — Aggregate Confusion Matrix (5-fold LOSO)")
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig2.tight_layout()
        fig2.savefig(FIGURES_DIR / f"{safe_name}_confusion_matrix_5fold.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)


# Missclassification summary for a single model, pooled across all folds, saved to CSV

def misclassification_summary(pooled, class_names, model_name="Attn (avg-anchor)"):
    if model_name not in pooled:
        print(f"WARNING: no pooled predictions for '{model_name}' — skipping misclassification summary.")
        return
    data = pooled[model_name]
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    wrong = y_true != y_pred

    summary = pd.DataFrame({
        "true": [class_names[i] for i in y_true[wrong]],
        "pred": [class_names[i] for i in y_pred[wrong]],
    }).value_counts().reset_index(name="count").sort_values("count", ascending=False)

    print(f"\n{model_name} — misclassification breakdown (all 5 folds pooled):")
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "misclassification_summary.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'misclassification_summary.csv'}")


# t-SNE visualization of raw concatenation vs. gated cross-attention latent space (Figure 3)

def run_tsne(X, perplexity=30, random_state=42):
    return TSNE(
        n_components=2, perplexity=min(perplexity, len(X) - 1),
        random_state=random_state, max_iter=1000, init='pca', learning_rate='auto',
    ).fit_transform(X)

# Runs the Wilcoxon signed-rank test and Cohen's d effect size for every model pair, returning matrices of p-values and d-values
def extract_latent(model, t, a, v, device, batch_size=64):
    model.eval()
    latents = []
    n = t.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            tb = torch.tensor(t[i:i + batch_size], dtype=torch.float32).to(device)
            ab = torch.tensor(a[i:i + batch_size], dtype=torch.float32).to(device)
            vb = torch.tensor(v[i:i + batch_size], dtype=torch.float32).to(device)

            t_feat = model.proj_t(tb).unsqueeze(1)
            a_feat = model.proj_a(ab).unsqueeze(1)
            v_feat = model.proj_v(vb).unsqueeze(1)

            if model.anchor == "text":
                q, kv1, kv2 = t_feat, a_feat, v_feat
            elif model.anchor == "audio":
                q, kv1, kv2 = a_feat, t_feat, v_feat
            elif model.anchor == "video":
                q, kv1, kv2 = v_feat, t_feat, a_feat
            else:  # avg
                w = torch.softmax(model.avg_weights, dim=0)
                q = w[0] * t_feat + w[1] * a_feat + w[2] * v_feat
                kv1, kv2 = t_feat, a_feat

            attn_out1, _ = model.attn_1(q, kv1, kv1)
            attn_out2, _ = model.attn_2(q, kv2, kv2)

            q_sq = q.squeeze(1)
            ctx1 = model.ln_1(attn_out1 + q).squeeze(1)
            ctx2 = model.ln_2(attn_out2 + q).squeeze(1)

            combined = torch.cat([q_sq, ctx1, ctx2], dim=1)
            gate_weights = torch.softmax(model.gate_layer(combined), dim=1)

            fused = torch.cat([
                gate_weights[:, 0:1] * q_sq,
                gate_weights[:, 1:2] * ctx1,
                gate_weights[:, 2:3] * ctx2,
            ], dim=1)
            latents.append(fused.cpu().numpy())

    return np.vstack(latents)


def plot_fusion_comparison_tsne(df_all, X_text, X_audio, X_video, y, le, anchor="avg", fold_session=TSNE_FOLD):
    split = get_fold_split(df_all, X_text, X_audio, X_video, y, fold_session, save_scalers=False)
    Xt_te, Xa_te, Xv_te, y_te = split["Xt_te"], split["Xa_te"], split["Xv_te"], split["y_te"]
    Xf_te = np.concatenate([Xt_te, Xa_te, Xv_te], axis=1)

    ckpt_path = CKPT_DIR / f"cross_attn_{anchor}anchor_fold{fold_session}.pt"
    if not ckpt_path.exists():
        print(f"WARNING: {ckpt_path} not found — run train_cross_attention.py first. Skipping Figure 3.")
        return

    with open(CKPT_DIR / "cross_attn_config.json") as f:
        cfg = json.load(f)
    model = CrossAttentionFusion(
        cfg["t_in"], cfg["a_in"], cfg["v_in"],
        embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"], dropout=cfg["dropout"], anchor=anchor,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)

    print(f"Extracting {anchor}-anchor fused latents (fold {fold_session} test set)...", end=" ", flush=True)
    Z_latent = extract_latent(model, Xt_te, Xa_te, Xv_te, DEVICE)
    print("done")

    emotion_labels = [le.classes_[i] for i in y_te]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.suptitle("t-SNE: Raw Concatenation vs. Gated Cross-Attention Latent Space", fontsize=13)

    for ax, (X_plot, title) in zip([ax1, ax2],
                                    [(Xf_te, "Early Fusion (raw concat)"),
                                     (Z_latent, f"Gated Cross-Attention latent ({anchor}-anchor)")]):
        print(f"t-SNE: {title}...", end=" ", flush=True)
        Z = run_tsne(X_plot)
        print("done")
        for emo in le.classes_:
            mask = np.array(emotion_labels) == emo
            ax.scatter(Z[mask, 0], Z[mask, 1], c=EMOTION_COLORS[emo], label=emo, alpha=0.55, s=18, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in le.classes_]
    fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_tsne_fusion_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'figure3_tsne_fusion_comparison.png'}")


# Raw concatenation vs. gated cross-attention latent space (Figure 3) — supplementary t-SNE panels for all three anchor variants

def plot_modality_tsne_panels(df_all, X_text, X_audio, X_video, y, le, fold_session=TSNE_FOLD):
    split = get_fold_split(df_all, X_text, X_audio, X_video, y, fold_session, save_scalers=False)
    Xt_te, Xa_te, Xv_te, y_te = split["Xt_te"], split["Xa_te"], split["Xv_te"], split["y_te"]
    Xf_te = np.concatenate([Xt_te, Xa_te, Xv_te], axis=1)

    tsne_sets = {
        "Text (RoBERTa)": Xt_te,
        "Audio (wav2vec2)": Xa_te,
        "Video (EfficientNet-B0)": Xv_te,
        "Early Fusion": Xf_te,
    }
    emotion_labels = [le.classes_[i] for i in y_te]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=150)
    fig.suptitle(f"t-SNE of Feature Spaces (Fold {fold_session} Test Set) — supplementary", fontsize=14, y=1.02)

    for ax, (title, X_plot) in zip(axes, tsne_sets.items()):
        print(f"Running t-SNE for {title}...", end=" ", flush=True)
        Z = run_tsne(X_plot)
        print("done")
        for emo in le.classes_:
            mask = np.array(emotion_labels) == emo
            ax.scatter(Z[mask, 0], Z[mask, 1], c=EMOTION_COLORS[emo], label=emo, alpha=0.55, s=18, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e) for e in le.classes_]
    fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supplementary_tsne_modalities.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'supplementary_tsne_modalities.png'}")


# Main

def main():
    df_all, X_text, X_audio, X_video, y, le = load_cached_features()
    class_names = list(le.classes_)

    pooled = load_pooled_predictions(class_names)

    plot_aggregate_confusion_matrices(pooled, class_names)
    misclassification_summary(pooled, class_names, model_name="Attn (avg-anchor)")
    plot_fusion_comparison_tsne(df_all, X_text, X_audio, X_video, y, le, anchor="avg", fold_session=TSNE_FOLD)
    plot_modality_tsne_panels(df_all, X_text, X_audio, X_video, y, le, fold_session=TSNE_FOLD)


if __name__ == "__main__":
    main()