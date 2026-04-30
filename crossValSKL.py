"""
Usage: python crossValSKL.py <CSVFile> <HyperparamFile.json> [<outputTreeFile>]

Same 10-fold CV grid search as crossVal.py, but uses sklearn's
DecisionTreeClassifier with criterion='entropy' (Information Gain only).
Saves the final tree as a text visualization if outputTreeFile is given.
"""
import sys
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OrdinalEncoder
from c45 import load_csv


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


def encode(train_X, test_X, cat_cols):
    if not cat_cols:
        return train_X, test_X
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    tr = train_X.copy()
    te = test_X.copy()
    tr[cat_cols] = enc.fit_transform(train_X[cat_cols])
    te[cat_cols] = enc.transform(test_X[cat_cols])
    return tr, te


def run_cv_skl(folds, col_types, class_col, feature_cols, threshold):
    cat_cols = [c for c in feature_cols if col_types.get(c, 0) > 0]
    all_preds, all_true = [], []

    for i in range(len(folds)):
        test = folds[i]
        train = pd.concat([folds[j] for j in range(len(folds)) if j != i],
                          ignore_index=True)
        X_train_enc, X_test_enc = encode(train[feature_cols], test[feature_cols], cat_cols)

        clf = DecisionTreeClassifier(
            criterion='entropy',
            min_impurity_decrease=threshold,
            random_state=42
        )
        clf.fit(X_train_enc, train[class_col])
        preds = clf.predict(X_test_enc).tolist()

        all_preds.extend(str(p) for p in preds)
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
        print("Usage: python crossValSKL.py <CSVFile> <HyperparamFile.json> [<outputTreeFile>]")
        sys.exit(1)

    csv_file = sys.argv[1]
    hp_file = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None

    df, col_types, class_col, feature_cols = load_csv(csv_file)
    cat_cols = [c for c in feature_cols if col_types.get(c, 0) > 0]

    with open(hp_file) as f:
        hparams = json.load(f)

    folds = k_fold_split(df)

    best_acc = -1.0
    best_thresh = best_preds = best_true = None

    for thresh in hparams.get("InfoGain", [0.0]):
        preds, true = run_cv_skl(folds, col_types, class_col, feature_cols, thresh)
        acc = accuracy(preds, true)
        print(f"InfoGain  threshold={thresh:.4f}  accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
            best_preds, best_true = preds, true

    print(f"\nBest Model: Splitting Metric=Information Gain, Threshold={best_thresh}")
    print(f"Cross-Validation Accuracy: {best_acc:.4f}")

    mat, classes = conf_matrix(best_preds, best_true)
    print_conf_matrix(mat, classes)

    if out_file:
        X = df[feature_cols].copy()
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[cat_cols] = enc.fit_transform(df[cat_cols])

        clf = DecisionTreeClassifier(
            criterion='entropy',
            min_impurity_decrease=best_thresh,
            random_state=42
        )
        clf.fit(X, df[class_col])
        tree_text = export_text(clf, feature_names=feature_cols)
        with open(out_file, 'w') as f:
            f.write(tree_text)
        print(f"\nFinal sklearn tree saved to {out_file}")


if __name__ == '__main__':
    main()
