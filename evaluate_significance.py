from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from loso_utils import ROOT

RESULTS_DIR = ROOT / "results"

RESULT_FILES = [
    "baselines_results.csv",
    "gcn_baselines_results.csv",
    "cross_attention_results.csv",
]

FOLD_COLS = ["S1", "S2", "S3", "S4", "S5"]

# Headline pairs to report in the paper (p-value and Cohen's d)
HEADLINE_PAIRS = [
    ("Attn (avg-anchor)", "MMGCN (avt)"),
    ("Attn (avg-anchor)", "DialogueGCN (text)"),
    ("Attn (avg-anchor)", "Attn (text-anchor)"),
]

# Merges results from all training scripts
def load_fold_results():
    fold_results = {}
    missing = []
    for fname in RESULT_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            missing.append(fname)
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            fold_results[row["Model"]] = [row[c] for c in FOLD_COLS]

    if missing:
        print(f"WARNING: missing results file(s), significance table will be incomplete: {missing}")
    if not fold_results:
        raise FileNotFoundError(
            "No results CSVs found. Run train_baselines.py, train_gcn_baselines.py, "
            "and train_cross_attention.py first."
        )
    return fold_results

# Wilcoxon signed-rank test for pairwise significance, and Cohen's d effect size
def pairwise_tests(fold_results):
    model_names = list(fold_results.keys())
    n = len(model_names)
    p_matrix = np.ones((n, n))
    d_matrix = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        a = np.array(fold_results[model_names[i]])
        b = np.array(fold_results[model_names[j]])
        diff = a - b

        try:
            _, p = stats.wilcoxon(diff, alternative='two-sided')
        except ValueError:
            p = 1.0  

        d = diff.mean() / (diff.std(ddof=1) + 1e-9)

        p_matrix[i, j] = p_matrix[j, i] = p
        d_matrix[i, j] = d
        d_matrix[j, i] = -d

    return model_names, p_matrix, d_matrix

def print_matrices(model_names, p_matrix, d_matrix):
    n = len(model_names)
    short = [nm.replace("Early Fusion", "EF")
             .replace("Late Fusion", "LF")
             .replace("(dynamic)", "(dyn)")
             .replace("-anchor", "")
             for nm in model_names]

    print("\n" + "=" * 70)
    print("PAIRWISE WILCOXON p-VALUES  (significant at p < 0.05 marked *)")
    print("=" * 70)
    header = f"{'':22}" + "".join(f"{s:>10}" for s in short)
    print(header)
    for i, name in enumerate(short):
        row = f"{name:22}"
        for j in range(n):
            if i == j:
                row += f"{'---':>10}"
            else:
                p = p_matrix[i, j]
                sig = "*" if p < 0.05 else " "
                row += f"{p:.3f}{sig:>4}"
        print(row)

    print("\n" + "=" * 70)
    print("COHEN'S d  (row vs col; positive = row is better)")
    print("=" * 70)
    print(header)
    for i, name in enumerate(short):
        row = f"{name:22}"
        for j in range(n):
            if i == j:
                row += f"{'---':>10}"
            else:
                row += f"{d_matrix[i, j]:>+9.2f} "
        print(row)

def print_significant_pairs(model_names, fold_results, p_matrix, d_matrix):
    print("\n--- Significant differences (p < 0.05) ---")
    found = False
    for i, j in combinations(range(len(model_names)), 2):
        p = p_matrix[i, j]
        if p < 0.05:
            a_mean = np.mean(fold_results[model_names[i]])
            b_mean = np.mean(fold_results[model_names[j]])
            winner = model_names[i] if a_mean > b_mean else model_names[j]
            loser = model_names[j] if a_mean > b_mean else model_names[i]
            print(f"  {winner:30s} > {loser:30s}  p={p:.4f}  d={d_matrix[i, j]:+.2f}")
            found = True
    if not found:
        print("  None at p < 0.05 (expected with only 5 folds -- report effect sizes; "
              "minimum achievable p-value with n=5 paired folds is 0.0625)")

def report_headline_pairs(model_names, p_matrix, d_matrix):
    print("\n--- Headline comparisons ---")
    for name_a, name_b in HEADLINE_PAIRS:
        if name_a not in model_names or name_b not in model_names:
            print(f"{name_a} vs {name_b}: SKIPPED (missing from results -- "
                  f"check all three training scripts have been run)")
            continue
        i, j = model_names.index(name_a), model_names.index(name_b)
        print(f"{name_a} vs {name_b}: p={p_matrix[i, j]:.4f}, d={d_matrix[i, j]:+.3f}")

def export_table4(model_names, p_matrix, d_matrix):
    sig_rows = []
    for i, j in combinations(range(len(model_names)), 2):
        sig_rows.append({
            "Model A": model_names[i],
            "Model B": model_names[j],
            "Wilcoxon p": p_matrix[i, j],
            "Cohen's d": d_matrix[i, j],
            "Significant (p<0.05)": p_matrix[i, j] < 0.05,
        })
    table4_df = pd.DataFrame(sig_rows).round(4)
    table4_df.to_csv(RESULTS_DIR / "table4_significance_full.csv", index=False)

    def is_headline(row):
        pair_fwd = (row["Model A"], row["Model B"])
        pair_rev = (row["Model B"], row["Model A"])
        return pair_fwd in HEADLINE_PAIRS or pair_rev in HEADLINE_PAIRS

    table4_headline = table4_df[table4_df.apply(is_headline, axis=1)]
    table4_headline.to_csv(RESULTS_DIR / "table4_significance_headline.csv", index=False)

    print(f"\nSaved: {RESULTS_DIR / 'table4_significance_full.csv'}")
    print(f"Saved: {RESULTS_DIR / 'table4_significance_headline.csv'}")
    return table4_headline

# Simple per-fold delta report for two models, to illustrate the effect size
def per_fold_delta(fold_results, name_a, name_b):
    if name_a not in fold_results or name_b not in fold_results:
        return
    a_scores, b_scores = fold_results[name_a], fold_results[name_b]
    print(f"\nPer-fold delta: {name_a} vs {name_b}")
    print(f"{'Session':<10} {name_b:>14} {name_a:>14} {'Delta':>8}")
    print("-" * 50)
    for s, (av, bv) in enumerate(zip(a_scores, b_scores), 1):
        delta = av - bv
        flag = "up" if delta > 0 else "down"
        print(f"  Ses0{s}     {bv:>14.3f} {av:>14.3f} {delta:>+7.3f} {flag}")
    mean_delta = np.mean(np.array(a_scores) - np.array(b_scores))
    print("-" * 50)
    print(f"  Mean      {np.mean(b_scores):>14.3f} {np.mean(a_scores):>14.3f} {mean_delta:>+7.3f}")


def main():
    fold_results = load_fold_results()
    print(f"Loaded fold scores for {len(fold_results)} models.")

    model_names, p_matrix, d_matrix = pairwise_tests(fold_results)
    print_matrices(model_names, p_matrix, d_matrix)
    print_significant_pairs(model_names, fold_results, p_matrix, d_matrix)
    report_headline_pairs(model_names, p_matrix, d_matrix)

    per_fold_delta(fold_results, "DialogueGCN (text)", "Text only")

    export_table4(model_names, p_matrix, d_matrix)


if __name__ == "__main__":
    main()