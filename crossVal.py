"""
Usage: python crossVal.py <CSVFile> <HyperparamFile.json> [<outputTreeFile>]

Performs 10-fold cross-validation grid search over the hyperparameters in
HyperparamFile.json, reports the best model's accuracy and confusion matrix,
and optionally trains a final tree on all data and saves it to outputTreeFile.

HyperparamFile.json format:
{
  "InfoGain": [0.0, 0.01, 0.05, ...],
  "Ratio":    [0.0, 0.01, 0.05, ...]
}
"""
import sys
import json
import pandas as pd
from c45 import c45, load_csv


def k_fold_split(df, k=10, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        folds.append(df.iloc[start:end].reset_index(drop=True))
    return folds


def run_cv(folds, col_types, class_col, feature_cols, metric, threshold):
    all_preds, all_true = [], []
    for i in range(len(folds)):
        test = folds[i]
        train = pd.concat([folds[j] for j in range(len(folds)) if j != i],
                          ignore_index=True)
        model = c45(metric=metric, threshold=threshold)
        model.fit(train[feature_cols], train[class_col], col_types=col_types)
        preds = model.predict(test[feature_cols])
        all_preds.extend(str(p) if p is not None else '' for p in preds)
        all_true.extend(str(v) for v in test[class_col])
    return all_preds, all_true


def accuracy(preds, true):
    return sum(p == t for p, t in zip(preds, true)) / len(true)


def conf_matrix(preds, true):
    classes = sorted(set(true))
    mat = {c: {c2: 0 for c2 in classes} for c in classes}
    for p, t in zip(preds, true):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat, classes


def print_conf_matrix(mat, classes):
    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print("\t" + "\t".join(classes))
    for c in classes:
        print(c + "\t" + "\t".join(str(mat[c].get(p, 0)) for p in classes))


def main():
    if len(sys.argv) < 3:
        print("Usage: python crossVal.py <CSVFile> <HyperparamFile.json> [<outputTreeFile>]")
        sys.exit(1)

    csv_file = sys.argv[1]
    hp_file = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None

    df, col_types, class_col, feature_cols = load_csv(csv_file)

    with open(hp_file) as f:
        hparams = json.load(f)

    folds = k_fold_split(df)

    best_acc = -1.0
    best_metric = best_thresh = best_preds = best_true = None

    for thresh in hparams.get("InfoGain", [0.0]):
        preds, true = run_cv(folds, col_types, class_col, feature_cols, 'gain', thresh)
        acc = accuracy(preds, true)
        print(f"InfoGain   threshold={thresh:.4f}  accuracy={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_metric, best_thresh = acc, 'gain', thresh
            best_preds, best_true = preds, true

    for thresh in hparams.get("Ratio", [0.0]):
        preds, true = run_cv(folds, col_types, class_col, feature_cols, 'gain_ratio', thresh)
        acc = accuracy(preds, true)
        print(f"GainRatio  threshold={thresh:.4f}  accuracy={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_metric, best_thresh = acc, 'gain_ratio', thresh
            best_preds, best_true = preds, true

    metric_name = "Information Gain" if best_metric == 'gain' else "Information Gain Ratio"
    print(f"\nBest Model: Splitting Metric={metric_name}, Threshold={best_thresh}")
    print(f"Cross-Validation Accuracy: {best_acc:.4f}")

    mat, classes = conf_matrix(best_preds, best_true)
    print_conf_matrix(mat, classes)

    if out_file:
        model = c45(metric=best_metric, threshold=best_thresh)
        model.fit(df[feature_cols], df[class_col],
                  dataset_name=csv_file, col_types=col_types)
        model.save_tree(out_file)
        print(f"\nFinal model tree saved to {out_file}")


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
