"""
Usage: python predict.py <CSVFile> <JSONFile> [eval]

Loads a saved JSON decision tree and makes predictions on the CSV data.
With the optional 'eval' argument, also prints accuracy stats and a
confusion matrix (requires the class column to be present in the CSV).
"""
import sys
from c45 import c45, load_csv


def print_confusion_matrix(matrix, classes):
    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print("\t" + "\t".join(classes))
    for actual in classes:
        print(actual + "\t" + "\t".join(str(matrix[actual].get(pred, 0)) for pred in classes))


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <CSVFile> <JSONFile> [eval]")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    do_eval = len(sys.argv) > 3 and sys.argv[3].lower() == 'eval'

    df, col_types, class_col, feature_cols = load_csv(csv_file)
    X_test = df[feature_cols]

    model = c45()
    model.col_types = col_types
    model.read_tree(json_file)

    raw_preds = model.predict(X_test)
    predictions = [str(p) if p is not None else 'UNKNOWN' for p in raw_preds]

    if not do_eval:
        for pred in predictions:
            print(pred)
        return

    y_true = [str(v) for v in df[class_col].tolist()]

    for i, (pred, actual) in enumerate(zip(predictions, y_true)):
        print(f"Record {i + 1}: Predicted={pred}, Actual={actual}")

    total = len(predictions)
    correct = sum(p == t for p, t in zip(predictions, y_true))
    incorrect = total - correct

    print(f"\nTotal records classified : {total}")
    print(f"Correctly classified     : {correct}")
    print(f"Incorrectly classified   : {incorrect}")
    print(f"Accuracy                 : {correct / total:.4f}")
    print(f"Error Rate               : {incorrect / total:.4f}")

    classes = sorted(set(y_true))
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    for pred, actual in zip(predictions, y_true):
        if actual in matrix and pred in classes:
            matrix[actual][pred] += 1

    print_confusion_matrix(matrix, classes)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
