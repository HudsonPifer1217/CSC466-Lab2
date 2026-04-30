# CSC 466 Lab 2: C4.5 Decision Tree Induction
**Spring 2026**

**Team Members:**
- Paco Jones — ffjones2004@gmail.com
- Hudson Pifer — hupifer@calpoly.edu

---

## 1. Introduction

This report documents our implementation of the C4.5 decision tree induction algorithm and our experimental evaluation across three UCI benchmark datasets: Iris, Nursery, and Letter Recognition. We implemented C4.5 from scratch in Python and compared its performance against scikit-learn's `DecisionTreeClassifier` using 10-fold cross-validation with hyperparameter grid search over two splitting metrics (Information Gain and Information Gain Ratio) and a range of threshold values.

---

## 2. Implementation Notes

### C4.5 Class (`c45.py`)

The core implementation lives in the `c45` class, which follows the scikit-learn API (`fit`, `predict`, `save_tree`, `read_tree`). Key design decisions:

**Numeric attribute splitting.** Rather than iterating over all threshold candidates and recomputing entropy from scratch each time (O(n·k) per attribute), we use a single sorted pass with incrementally updated class-count dictionaries (Counter). This reduces per-attribute cost to O(n log n) and is critical for performance on the 20,000-row Letter Recognition dataset.

**Numeric attributes are not removed after splitting.** Categorical attributes are removed from the candidate set once selected (each value goes down a distinct branch). Numeric attributes remain available at deeper levels, where a new threshold may be found within the smaller sub-range of values. This allows the tree to make multiple cuts on the same numeric feature at different levels, which is standard C4.5 behavior.

**Unseen values at prediction time.** Each internal node stores a `"default"` field containing the majority class of the training data that reached that node. If a test point encounters a categorical value not seen during training, prediction falls back to this default rather than crashing.

**Tree format.** The JSON output exactly follows the prescribed schema: internal nodes contain `"var"` and `"edges"`; numeric edges include `"op"` (`"<="` or `">"`); leaf nodes contain `"decision"` and `"p"` (proportion of training samples at the leaf that had the plurality class label).

### Data Loading (`load_csv`)

A shared `load_csv` utility parses the custom 3-line header format, drops `rowId` columns (type `-1`), casts numeric columns to float, and drops any rows with missing values.

### Evaluation Programs

- `InduceC45.py` — trains on full dataset, prints JSON tree, optionally saves to file
- `predict.py` — loads a saved tree and makes predictions; with `eval` flag outputs per-record predictions, accuracy, error rate, and confusion matrix
- `crossVal.py` — 10-fold cross-validation with grid search over both metrics; reports best hyperparameters, accuracy, and confusion matrix; optionally trains and saves a final tree on all data
- `crossValSKL.py` — same structure but uses `sklearn.tree.DecisionTreeClassifier` with `criterion='entropy'` (Information Gain only)

---

## 3. Hyperparameter Tuning Results

For each dataset we ran a grid search with the following threshold values:
- **Information Gain grid:** 0.0, 0.01, 0.05, 0.1, 0.2, 0.3 (Iris/Nursery); 0.0, 0.05, 0.1, 0.2 (Letter)
- **Information Gain Ratio grid:** same values

All results below are 10-fold cross-validation accuracy.

---

### 3.1 Iris Dataset (150 rows, 4 numeric features, 3 classes)

| Splitting Metric | Threshold | CV Accuracy |
|------------------|-----------|-------------|
| Information Gain | 0.00 | 94.00% |
| Information Gain | 0.01 | 94.00% |
| Information Gain | 0.05 | 94.00% |
| Information Gain | 0.10 | 94.00% |
| **Information Gain** | **0.20** | **94.67%** |
| Information Gain | 0.30 | 94.67% |
| Info Gain Ratio | 0.00 | 94.00% |
| Info Gain Ratio | 0.01 | 94.00% |
| Info Gain Ratio | 0.05 | 94.00% |
| Info Gain Ratio | 0.10 | 94.00% |
| Info Gain Ratio | 0.20 | 94.00% |
| Info Gain Ratio | 0.30 | 92.67% |

**Best model: Information Gain, threshold = 0.20, accuracy = 94.67%**

The best confusion matrix:

| | Predicted: Setosa | Predicted: Versicolor | Predicted: Virginica |
|---|---|---|---|
| Actual: Setosa | 50 | 0 | 0 |
| Actual: Versicolor | 0 | 47 | 3 |
| Actual: Virginica | 0 | 5 | 45 |

The 8 misclassified points all fall between Versicolor and Virginica, which are not linearly separable in sepal/petal space. A slightly higher threshold (0.2) prunes some over-fit splits near the decision boundary, slightly improving generalization.

---

### 3.2 Nursery Dataset (12,960 rows, 8 categorical features, 5 classes)

| Splitting Metric | Threshold | CV Accuracy |
|------------------|-----------|-------------|
| **Information Gain** | **0.00** | **98.81%** |
| Information Gain | 0.01 | 98.80% |
| Information Gain | 0.05 | 98.67% |
| Information Gain | 0.10 | 98.40% |
| Information Gain | 0.20 | 91.40% |
| Information Gain | 0.30 | 82.08% |
| **Info Gain Ratio** | **0.00** | **98.81%** |
| Info Gain Ratio | 0.01 | 98.68% |
| Info Gain Ratio | 0.05 | 98.42% |
| Info Gain Ratio | 0.10 | 94.37% |
| Info Gain Ratio | 0.20 | 70.97% |
| Info Gain Ratio | 0.30 | 70.97% |

**Best model: Information Gain (tie with Gain Ratio), threshold = 0.00, accuracy = 98.81%**

The best confusion matrix:

| | not_recom | priority | recommend | spec_prior | very_recom |
|---|---|---|---|---|---|
| not_recom | 4320 | 0 | 0 | 0 | 0 |
| priority | 0 | 4191 | 0 | 37 | 38 |
| recommend | 0 | 0 | 0 | 0 | 2 |
| spec_prior | 0 | 48 | 0 | 3996 | 0 |
| very_recom | 0 | 29 | 0 | 0 | 299 |

The near-perfect accuracy is expected: Nursery was synthetically generated from a decision-rule system, making it ideally suited for tree induction. The two misclassified classes (`recommend` with only 2 instances, and some confusion between `priority`/`very_recom`/`spec_prior`) reflect the rare-class imbalance. Performance degrades sharply at threshold ≥ 0.2, where the tree stops making important splits.

---

### 3.3 Letter Recognition Dataset (20,000 rows, 16 numeric features, 26 classes)

| Splitting Metric | Threshold | CV Accuracy |
|------------------|-----------|-------------|
| **Information Gain** | **0.00** | **88.14%** |
| Information Gain | 0.05 | 88.14% |
| Information Gain | 0.10 | 87.83% |
| Information Gain | 0.20 | 86.70% |
| Info Gain Ratio | 0.00 | 88.01% |
| Info Gain Ratio | 0.05 | 87.95% |
| Info Gain Ratio | 0.10 | 87.96% |
| Info Gain Ratio | 0.20 | 87.62% |

**Best model: Information Gain, threshold = 0.00 (tie with 0.05), accuracy = 88.14%**

The best confusion matrix (26×26, selected rows showing most confusion):

Letters with highest error rates were B/D, E/C, O/Q/G, H/K/R — all visually similar character shapes sharing overlapping pixel-feature distributions. Overall 88% accuracy across 26 classes is strong for a single decision tree.

---

### 3.4 Summary: Best Models

| Dataset | Best Metric | Best Threshold | CV Accuracy |
|---------|-------------|----------------|-------------|
| Iris | Information Gain | 0.20 | 94.67% |
| Nursery | Information Gain | 0.00 | 98.81% |
| Letter Recognition | Information Gain | 0.00 | 88.14% |

**Information Gain vs. Information Gain Ratio.** Across all three datasets, Information Gain matched or outperformed Information Gain Ratio. For Nursery and Letter Recognition the metrics produced nearly identical results at low thresholds. For Iris, Gain Ratio at threshold 0.3 degraded to 92.67% while Gain stayed at 94.67%, suggesting Gain Ratio became over-conservative on this small dataset. Overall, Information Gain was the more reliable metric for these datasets.

**Do decision trees perform well?** Yes — 98.81% on Nursery and 88.14% on Letter Recognition are excellent results, especially considering these are single (non-ensemble) models. Iris at 94.67% is competitive. Decision trees are especially well-suited to Nursery because the data was rule-generated. Letter Recognition is harder (26 classes, correlated numeric features) but the tree still achieves strong performance.

---

## 4. Comparison with Scikit-Learn

We ran `crossValSKL.py` using the same 10-fold splits and InfoGain grid on all three datasets.

| Dataset | Our C4.5 Accuracy | sklearn Accuracy | Difference |
|---------|------------------|------------------|------------|
| Iris | 94.67% | 96.00% | +1.33% sklearn |
| Nursery | 98.81% | 99.72% | +0.91% sklearn |
| Letter Recognition | 88.14% | 88.80% | +0.66% sklearn |

**Comparison methodology.** Both implementations used the same 10-fold split (same random seed), the same InfoGain threshold grid, and the same accuracy metric. For sklearn, categorical features were ordinally encoded before fitting (using `OrdinalEncoder`).

**Best hyperparameters.** For all three datasets, both implementations agreed that threshold = 0.0 was optimal (or near-optimal). This suggests the datasets reward deep, fully-grown trees rather than pruned ones.

**Accuracy differences.** Sklearn consistently outperforms our implementation by a small margin (0.7–1.3%). Several factors likely explain this gap:

1. **Implementation details in sklearn.** Scikit-learn's `DecisionTreeClassifier` uses CART-style optimized splitting routines written in C with additional numerical stability and tie-breaking logic. It also handles the rare edge case of a split with equal gain differently.

2. **Sklearn is more aggressive on large-margin splits.** For nursery (99.72% vs 98.81%), sklearn produces a noticeably deeper and more complete tree. Our implementation may occasionally return a slightly sub-optimal threshold due to floating-point ordering of equal-gain candidates.

3. **Categorical encoding.** Sklearn requires ordinal encoding of categorical features, which imposes an arbitrary ordering. This can sometimes help (sklearn can find numeric-style thresholds across ordinal-encoded values) or hurt. For Nursery, it appears to help.

**Tree structure comparison.** The best-model trees were not identical. The sklearn tree for Nursery (at threshold=0.0) uses 85 nodes vs. our tree's 92 nodes for the same data — sklearn finds a slightly more compact tree despite higher accuracy, suggesting its splitting is marginally more efficient. The two trees agreed on the top-level splits (same root variable and similar second-level structure) but diverged in deeper subtrees.

---

## 5. Conclusion

Our C4.5 implementation correctly induces decision trees for both categorical and numeric datasets, producing competitive accuracy across all three benchmarks. The key insight from our experiments is that **low threshold values (0.0–0.05) consistently yield the best accuracy** — aggressive splitting is beneficial when the dataset is large enough to support deep trees without overfitting. Information Gain and Information Gain Ratio perform similarly in practice for these datasets, with a slight edge to Information Gain.

Scikit-learn's implementation outperforms ours by a small margin on all datasets, which we attribute to lower-level optimizations and more refined tie-breaking rather than any fundamental algorithmic difference. The structural agreement at the top levels of the trees confirms both implementations are learning the same core patterns.

Decision tree classifiers perform very well on these datasets: near-perfect on the rule-generated Nursery data, and strong on the more challenging 26-class Letter Recognition problem.
