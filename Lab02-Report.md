# CSC 466 Lab 2: C4.5 Decision Tree Induction
Spring 2026

Team Members:
- Paco Jones -- pjones16@calpoly.edu
- Hudson Pifer -- hupifer@calpoly.edu

---

## 1. Introduction

This report documents our implementation of the C4.5 decision tree induction algorithm and our experimental evaluation across three UCI benchmark datasets: Iris, Nursery, and Letter Recognition. We implemented C4.5 from scratch in Python and compared its performance against scikit-learn's DecisionTreeClassifier using 10-fold cross-validation with hyperparameter grid search over two splitting metrics (Information Gain and Information Gain Ratio) and a range of threshold values.

---

## 2. Hyperparameter Tuning Results

For each dataset we ran a grid search with the following threshold values:
- Information Gain grid: 0.0, 0.01, 0.05, 0.1, 0.2, 0.3 (Iris/Nursery); 0.0, 0.05, 0.1, 0.2 (Letter)
- Information Gain Ratio grid: same values

All results below are 10-fold cross-validation accuracy.

---

### 2.1 Iris Dataset

| Splitting Metric | Threshold | CV Accuracy |
|------------------|-----------|-------------|
| Information Gain | 0.00 | 94.00% |
| Information Gain | 0.01 | 94.00% |
| Information Gain | 0.05 | 94.00% |
| Information Gain | 0.10 | 94.00% |
| Information Gain | 0.20 | 94.67% |
| Information Gain | 0.30 | 94.67% |
| Info Gain Ratio | 0.00 | 94.00% |
| Info Gain Ratio | 0.01 | 94.00% |
| Info Gain Ratio | 0.05 | 94.00% |
| Info Gain Ratio | 0.10 | 94.00% |
| Info Gain Ratio | 0.20 | 94.00% |
| Info Gain Ratio | 0.30 | 92.67% |

Best model: Information Gain, threshold = 0.20, accuracy = 94.67%

The best confusion matrix:

| | Predicted: Setosa | Predicted: Versicolor | Predicted: Virginica |
|---|---|---|---|
| Actual: Setosa | 50 | 0 | 0 |
| Actual: Versicolor | 0 | 47 | 3 |
| Actual: Virginica | 0 | 5 | 45 |

---

### 2.2 Nursery Dataset

| Splitting Metric | Threshold | CV Accuracy |
|------------------|-----------|-------------|
| Information Gain | 0.00 | 98.81% |
| Information Gain | 0.01 | 98.80% |
| Information Gain | 0.05 | 98.67% |
| Information Gain | 0.10 | 98.40% |
| Information Gain | 0.20 | 91.40% |
| Information Gain | 0.30 | 82.08% |
| Info Gain Ratio | 0.00 | 98.81% |
| Info Gain Ratio | 0.01 | 98.68% |
| Info Gain Ratio | 0.05 | 98.42% |
| Info Gain Ratio | 0.10 | 94.37% |
| Info Gain Ratio | 0.20 | 70.97% |
| Info Gain Ratio | 0.30 | 70.97% |

Best model: Information Gain (tie with Gain Ratio), threshold = 0.00, accuracy = 98.81%

The best confusion matrix:

| | not_recom | priority | recommend | spec_prior | very_recom |
|---|---|---|---|---|---|
| not_recom | 4320 | 0 | 0 | 0 | 0 |
| priority | 0 | 4191 | 0 | 37 | 38 |
| recommend | 0 | 0 | 0 | 0 | 2 |
| spec_prior | 0 | 48 | 0 | 3996 | 0 |
| very_recom | 0 | 29 | 0 | 0 | 299 |

---

### 2.3 Letter Recognition Dataset

| Splitting Metric | Threshold | CV Accuracy |
|------------------|-----------|-------------|
| Information Gain | 0.00 | 88.14% |
| Information Gain | 0.05 | 88.14% |
| Information Gain | 0.10 | 87.83% |
| Information Gain | 0.20 | 86.70% |
| Info Gain Ratio | 0.00 | 88.01% |
| Info Gain Ratio | 0.05 | 87.95% |
| Info Gain Ratio | 0.10 | 87.96% |
| Info Gain Ratio | 0.20 | 87.62% |

Best model: Information Gain, threshold = 0.00 (tie with 0.05), accuracy = 88.14%

---

### 2.4 Summary: Best Models

| Dataset | Best Metric | Best Threshold | CV Accuracy |
|---------|-------------|----------------|-------------|
| Iris | Information Gain | 0.20 | 94.67% |
| Nursery | Information Gain | 0.00 | 98.81% |
| Letter Recognition | Information Gain | 0.00 | 88.14% |

Information Gain vs. Information Gain Ratio. Across all three datasets, Information Gain matched or outperformed Information Gain Ratio. For Nursery and Letter Recognition the metrics produced nearly identical results at low thresholds. For Iris, Gain Ratio at threshold 0.3 degraded to 92.67% while Gain stayed at 94.67%, suggesting Gain Ratio became over-conservative on this small dataset. Overall, Information Gain was the more reliable metric for these datasets.

Do decision trees perform well? Yes -- 98.81% on Nursery and 88.14% on Letter Recognition are excellent results, especially considering these are non-ensemble models.

---

## 3. Comparison with Scikit-Learn

We ran crossValSKL.py using the same 10-fold splits and InfoGain grid on all three datasets.

| Dataset | Our C4.5 Accuracy | sklearn Accuracy | Difference |
|---------|------------------|------------------|------------|
| Iris | 94.67% | 96.00% | +1.33% sklearn |
| Nursery | 98.81% | 99.72% | +0.91% sklearn |
| Letter Recognition | 88.14% | 88.80% | +0.66% sklearn |

Comparison methodology. Both implementations used the same 10-fold split (same random seed), the same InfoGain threshold grid, and the same accuracy metric. For sklearn, categorical features were ordinally encoded before fitting (using OrdinalEncoder).

Best hyperparameters. For all three datasets, both implementations agreed that threshold = 0.0 was optimal (or near-optimal). This suggests the datasets reward deep, fully-grown trees rather than pruned ones.

Accuracy differences. Sklearn consistently outperforms our implementation by a small margin (0.7-1.3%).

---

## 4. Conclusion

Our C4.5 implementation correctly induces decision trees for both categorical and numeric datasets, producing competitive accuracy across all three benchmarks. The key insight from our experiments is that low threshold values (0.0-0.05) consistently yield the best accuracy -- aggressive splitting is beneficial when the dataset is large enough to support deep trees without overfitting. Information Gain and Information Gain Ratio perform similarly in practice for these datasets, with a slight edge to Information Gain.

Scikit-learn's implementation outperforms ours by a small margin on all datasets.

Decision tree classifiers perform very well on these datasets: near-perfect on the rule-generated Nursery data, and strong on the more challenging 26-class Letter Recognition problem.
