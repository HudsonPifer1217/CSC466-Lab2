import math
import json
import sys
import pandas as pd
from collections import Counter
from io import StringIO


def load_csv(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    col_names = [c.strip() for c in lines[0].strip().split(',')]
    col_type_vals = [int(x.strip()) for x in lines[1].strip().split(',')]
    class_col = lines[2].strip()
    col_types = dict(zip(col_names, col_type_vals))

    data_str = ''.join(lines[3:])
    df = pd.read_csv(StringIO(data_str), header=None, names=col_names)

    id_cols = [c for c, t in col_types.items() if t == -1]
    df = df.drop(columns=id_cols, errors='ignore')
    col_types = {c: t for c, t in col_types.items() if t != -1}

    for col, t in col_types.items():
        if t == 0:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != class_col]

    return df, col_types, class_col, feature_cols


class c45:
    def __init__(self, metric='gain_ratio', threshold=0.0):
        self.metric = metric
        self.threshold = threshold
        self.tree = None
        self.col_types = {}
        self.class_col = ''
        self.dataset_name = ''

    def _entropy_counts(self, counts, n):
        if n == 0:
            return 0.0
        return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    def _entropy_series(self, labels):
        n = len(labels)
        if n == 0:
            return 0.0
        return self._entropy_counts(Counter(labels), n)

    def _numeric_best_split(self, data, attr, class_col):
        subset = data[[attr, class_col]].sort_values(attr)
        vals = subset[attr].values
        labels = subset[class_col].values
        n = len(vals)
        if n <= 1:
            return 0.0, None

        base_ent = self._entropy_series(data[class_col])
        right_counts = Counter(labels)
        left_counts = Counter()
        left_n, right_n = 0, n
        best_gain, best_t = -1.0, None

        for i in range(n - 1):
            lbl = labels[i]
            left_counts[lbl] += 1
            right_counts[lbl] -= 1
            if right_counts[lbl] == 0:
                del right_counts[lbl]
            left_n += 1
            right_n -= 1

            if vals[i] == vals[i + 1]:
                continue

            left_ent = self._entropy_counts(left_counts, left_n)
            right_ent = self._entropy_counts(right_counts, right_n)
            gain = base_ent - (left_n / n * left_ent + right_n / n * right_ent)

            if gain > best_gain:
                best_gain = gain
                best_t = float((vals[i] + vals[i + 1]) / 2.0)

        return (best_gain if best_gain > 0 else 0.0), best_t

    def _score(self, data, attr, class_col):
        n = len(data)
        is_numeric = self.col_types.get(attr, 0) == 0

        if is_numeric:
            gain, t = self._numeric_best_split(data, attr, class_col)
            if t is None:
                return 0.0, None
            if self.metric == 'gain_ratio':
                left_n = int((data[attr] <= t).sum())
                right_n = n - left_n
                if left_n > 0 and right_n > 0:
                    sl, sr = left_n / n, right_n / n
                    si = -(sl * math.log2(sl) + sr * math.log2(sr))
                    return (gain / si if si > 0 else 0.0), t
            return gain, t
        else:
            base_ent = self._entropy_series(data[class_col])
            weighted, part_sizes = 0.0, []
            for _, grp in data.groupby(attr)[class_col]:
                weighted += (len(grp) / n) * self._entropy_series(grp)
                part_sizes.append(len(grp))
            gain = base_ent - weighted
            if self.metric == 'gain_ratio':
                si = -sum((p / n) * math.log2(p / n) for p in part_sizes if p > 0)
                return (gain / si if si > 0 else 0.0), None
            return gain, None

    def _best_attribute(self, data, attributes, class_col):
        best_attr, best_score, best_t = None, -1.0, None
        for attr in attributes:
            score, t = self._score(data, attr, class_col)
            if score > best_score:
                best_attr, best_score, best_t = attr, score, t
        return best_attr, best_score, best_t

    def _majority(self, data, class_col):
        counts = data[class_col].value_counts()
        return str(counts.index[0]), float(counts.iloc[0] / len(data))

    def _build(self, data, attributes, class_col, default):
        if len(data) == 0:
            return {"leaf": {"decision": str(default), "p": 0.0}}

        maj, maj_p = self._majority(data, class_col)

        if data[class_col].nunique() == 1:
            return {"leaf": {"decision": maj, "p": 1.0}}

        if not attributes:
            return {"leaf": {"decision": maj, "p": maj_p}}

        best_attr, best_score, best_t = self._best_attribute(data, attributes, class_col)

        if best_attr is None or best_score <= self.threshold:
            return {"leaf": {"decision": maj, "p": maj_p}}

        is_numeric = self.col_types.get(best_attr, 0) == 0
        node = {"var": best_attr, "default": maj, "edges": []}

        if is_numeric:
            remaining = list(attributes)
            for subset, op in [(data[data[best_attr] <= best_t], "<="),
                               (data[data[best_attr] > best_t], ">")]:
                child = self._build(subset, remaining, class_col, maj)
                edge = {"value": best_t, "op": op}
                edge["leaf" if "leaf" in child else "node"] = child.get("leaf", child)
                node["edges"].append({"edge": edge})
        else:
            remaining = [a for a in attributes if a != best_attr]
            for val in data[best_attr].unique():
                subset = data[data[best_attr] == val]
                child = self._build(subset, remaining, class_col, maj)
                edge = {"value": str(val)}
                edge["leaf" if "leaf" in child else "node"] = child.get("leaf", child)
                node["edges"].append({"edge": edge})

        return node

    def fit(self, X, y, dataset_name='', col_types=None):
        if col_types is not None:
            self.col_types = col_types
        self.dataset_name = dataset_name

        class_col = (y.name if hasattr(y, 'name') and y.name is not None else 'class')
        self.class_col = class_col

        data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        data[class_col] = y.values if hasattr(y, 'values') else list(y)

        attributes = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        default = str(data[class_col].mode()[0])

        root = self._build(data, attributes, class_col, default)
        self.tree = {"dataset": dataset_name, "node": root}
        return self.tree

    def _traverse(self, row, node):
        if "leaf" in node:
            return node["leaf"]["decision"]

        var = node["var"]
        val = row[var]
        is_numeric = bool(node["edges"]) and "op" in node["edges"][0]["edge"]

        for ew in node["edges"]:
            edge = ew["edge"]
            ev = edge["value"]
            if is_numeric:
                op = edge["op"]
                match = (op == "<=" and val <= ev) or (op == ">" and val > ev)
            else:
                match = str(val) == str(ev)

            if match:
                if "node" in edge:
                    return self._traverse(row, edge["node"])
                return edge["leaf"]["decision"]

        return node.get("default")

    def predict(self, X_test):
        if self.tree is None:
            raise ValueError("No tree. Call fit() or read_tree() first.")
        root = self.tree["node"]
        if isinstance(X_test, pd.DataFrame):
            return [self._traverse(row, root) for _, row in X_test.iterrows()]
        return [self._traverse(dict(zip(range(len(r)), r)), root) for r in X_test]

    def save_tree(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.tree, f, indent=2)

    def read_tree(self, filename):
        with open(filename, 'r') as f:
            self.tree = json.load(f)
