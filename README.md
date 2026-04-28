# ECE-2028-Final-Project-Fairness-Audit
# Fairness Audit: UCI Adult Income Classifier

A Jupyter notebook conducting a fairness audit of a binary classifier trained on the UCI Adult income dataset, structured around the **Map**, **Measure**, and **Manage** functions of the NIST AI Risk Management Framework. The audit also examines a parallel governance question: who should be responsible for conducting the audit (internal developers, external independent auditors, or a hybrid of both).

## Contents

The notebook (`Fairness_Audit_Final_Project.ipynb`) is organized into five sections:

1. **Setup** - imports and configuration
2. **Map** - system context, stakeholder map, dataset profiling, and base-rate analysis by sex
3. **Measure** - baseline classifier training, disaggregated performance metrics (Fairlearn), and group-fairness metrics (AIF360)
4. **Manage** - two mitigation strategies (Reweighing and Threshold Optimization) with side-by-side comparison of accuracy and fairness tradeoffs
5. **Auditor Self-Assessment** - a decision tool that scores any AI system across five dimensions and recommends an internal-only, hybrid, or full external audit

## Dependencies

The notebook was developed and tested with Python 3.9–3.11. It requires the following packages:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data handling |
| `matplotlib` | Plotting |
| `seaborn` | Plot styling |
| `scikit-learn` | Logistic regression baseline classifier |
| `fairlearn` | Disaggregated metrics (`MetricFrame`) and threshold-based postprocessing (`ThresholdOptimizer`) |
| `aif360` | UCI Adult dataset loader, group-fairness metrics (`ClassificationMetric`), and the Reweighing preprocessing algorithm |

## Setup Instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv fairness-audit
source fairness-audit/bin/activate     # macOS/Linux
fairness-audit\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn fairlearn aif360
```

If you encounter dependency conflicts with AIF360, install with the additional Reductions extra:

```bash
pip install 'aif360[Reductions]'
```

### 3. Download the UCI Adult dataset

AIF360 does not bundle the raw data. You must place the three required files in AIF360's expected directory before running the notebook.

**On the first run**, AIF360 will print the exact local path it expects. The location is typically:

```
<your_python_env>/lib/python3.x/site-packages/aif360/data/raw/adult/
```

Download the following three files from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/2/adult) and place them in that directory:

- `adult.data`
- `adult.test`
- `adult.names`

Once these files are in place, the `AdultDataset()` loader in the notebook will work without further configuration.

### 4. Launch the notebook

```bash
jupyter notebook Fairness_Audit_Final_Project.ipynb
```

## Reproducing the Reported Results

All results in the notebook are deterministic and reproducible because:

- The random seed is fixed (`np.random.seed(42)` at the top of the notebook, and `random_state=42` in every `train_test_split` and `LogisticRegression` call).
- The AIF360 dataset split uses `seed=42`.
- No stochastic mitigations are used (Reweighing is deterministic given the data; ThresholdOptimizer is deterministic given the trained classifier).

To reproduce all reported metrics, plots, and the final audit recommendation, simply run the notebook cells in order from top to bottom (`Kernel → Restart & Run All` in Jupyter).

The expected baseline results are:

- Overall accuracy: **0.847**
- Disparate impact (baseline): **0.318**
- Disparate impact after Reweighing: **0.641**
- Disparate impact after Threshold Optimization: **1.066**
- Auditor self-assessment score for the loan-screening scenario: **11/15 (full external audit recommended)**

If your numbers differ slightly, confirm that:

1. You are using the same versions of `scikit-learn`, `fairlearn`, and `aif360` (the fairness libraries are still under active development and metric implementations occasionally change between versions).
2. The UCI Adult data files in your AIF360 directory match the canonical versions linked above.

## Datasets and References

All datasets, frameworks, and toolkits used in this audit are linked in the **References** section at the end of the notebook (final cell).

## Repository Structure

```
.
├── Fairness_Audit_Final_Project.ipynb     # main audit notebook
└── README.md                              # this file
```

## License and Attribution

This audit is based on the open-source toolkits **Fairlearn** (Microsoft Research) and **AIF360** (IBM Research). See the References section of the notebook for full citations.
