"""
Usage: python InduceC45.py <TrainingSetFile.csv> [<fileToSave>]

Trains a C4.5 decision tree on the full CSV file and prints the JSON tree
to stdout. Optionally saves the JSON to a file.
"""
import sys
import json
from c45 import c45, load_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python InduceC45.py <TrainingSetFile.csv> [<fileToSave>]")
        sys.exit(1)

    csv_file = sys.argv[1]
    save_file = sys.argv[2] if len(sys.argv) > 2 else None

    df, col_types, class_col, feature_cols = load_csv(csv_file)
    X = df[feature_cols]
    y = df[class_col]

    model = c45()
    tree = model.fit(X, y, dataset_name=csv_file, col_types=col_types)

    print(json.dumps(tree, indent=2))

    if save_file:
        model.save_tree(save_file)
        print(f"Tree saved to {save_file}", file=sys.stderr)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
