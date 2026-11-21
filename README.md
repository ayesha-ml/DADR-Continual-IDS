# DADR: Drift-Aware Dual-Pool Experience Replay for Continual Network Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-green.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

**DADR (Drift-Aware Dual-Pool Experience Replay)** is a memory-bounded continual learning approach for network intrusion detection. It addresses the challenges of changing network traffic, class imbalance, and the arrival of new attack classes.

DADR uses two separate replay pools: a class-balanced attack pool and a normal-traffic pool. A Population Stability Index (PSI) detector monitors changes in normal-traffic feature distributions and triggers a partial refresh of the normal pool when drift is detected.

The framework is evaluated on CIC-IDS2017 using chronological traffic streams, two evaluation scenarios, multiple replay and continual learning baselines, and three random seeds.

---

## 1. Key Features

- **Dual-pool replay memory:** Divides a fixed memory budget between attack and normal traffic, using a 70:30 allocation.
- **Class-balanced attack memory:** Maintains separate memory allocations for observed attack classes.
- **Normal-traffic reservoir sampling:** Uses reservoir sampling to maintain a representative sample of normal traffic.
- **PSI-based drift detection:** Monitors changes in normal-traffic feature distributions.
- **Partial memory refresh:** Replaces up to 35% of the normal-pool capacity when drift is detected, subject to the availability of incoming normal samples.
- **Dynamic class expansion:** Extends the classifier output layer when new attack classes appear.
- **Multiple baselines:** Compares DADR with fine-tuning, regularization, replay-based methods, and an offline Joint training reference.

## 2. Architecture Overview

DADR consists of three main components:

```text
              Incoming Traffic Window (W_t)
                           |
             +-------------+-------------+
             |                           |
             v                           v
       PSI Drift Detector          Class Separation
             |                    (Normal / Attack)
             |                           |
       Drift Detected?                    |
             |                           |
             v                           v
       Refresh Normal              Dual-Pool Memory
       Replay Pool                       |
                                  +------+------+
                                  |             |
                                  v             v
                             Attack Pool   Normal Pool
                              (70%)          (30%)
                                  |             |
                           Class-Balanced   Reservoir
                              Sampling       Sampling
                                  |             |
                                  +------+------+
                                         |
                                         v
                               Replay + Current
                                  Traffic Window
                                         |
                                         v
                                  Continual MLP
                                  Classifier
                                         |
                                         v
                                  Updated Model
```

The attack pool maintains class-balanced samples from observed attack classes. The normal pool stores benign traffic and is monitored for distribution changes.

The PSI detector calculates changes in normal-traffic feature distributions. When the threshold is reached, the normal pool is partially refreshed using incoming normal samples.

## 3. Repository Structure

```text
DADR-Continual-IDS/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── dadr_buffer.py
│   ├── drift_detector.py
│   ├── baselines.py
│   ├── trainer.py
│   └── experiments.py
│
├── scripts/
│   ├── run_main_benchmark.py
│   └── plot_results.py
│
├── results/
│   ├── main_benchmark_summary.csv
│   ├── main_benchmark_windows.csv
│   ├── ablation_psi.csv
│   └── figures/
│
├── requirements.txt
└── README.md
```

### Main source files

| File | Description |
|---|---|
| `config.py` | Experimental settings, paths, seeds, and scenario configuration |
| `data_loader.py` | CIC-IDS2017 preprocessing and chronological stream preparation |
| `model.py` | MLP classifier and dynamic output-layer expansion |
| `dadr_buffer.py` | Dual-pool replay memory management |
| `drift_detector.py` | PSI-based distribution shift detection |
| `baselines.py` | Implementations of the comparison methods |
| `trainer.py` | Continual training and evaluation across stream windows |
| `experiments.py` | Experiment execution and evaluation support |

## 4. Dataset and Experimental Setup

### Dataset

The experiments use the **CIC-IDS2017** dataset provided by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The dataset contains benign traffic and multiple attack types collected over five days. The experiments use the eight official MachineLearningCSV files.

The preprocessing pipeline includes:

- Combining the dataset files and organizing traffic chronologically.
- Cleaning column names and identifying the target label.
- Normalizing benign and normal labels.
- Removing configured identifier columns.
- Retaining numeric features.
- Replacing infinite values with missing values and removing rows with missing numeric features.
- Converting feature values to `float32`.
- Encoding normal traffic as class 0 and assigning sorted labels to attack classes.
- Fitting the StandardScaler on the initial training period only.

### Stream Construction

The dataset is divided chronologically into an initial training period and a sequence of traffic windows.

| Stage | Portion |
|---|---:|
| Initial training period (T0) | 30% |
| Remaining streaming data | 70% |
| Number of sequential stream windows | 10 |

The model is initially trained on T0. The remaining traffic is then processed one window at a time.

For each incoming window, the model first makes predictions before adapting to that window. It is then trained using the current traffic window and stored replay samples. The replay memory is updated as part of the continual learning process.

### Evaluation Scenarios

**Scenario S1: Original Chronological Stream**

Uses the original chronological traffic stream. It evaluates how the methods handle sequentially arriving attack classes and whether previously learned information is retained.

**Scenario S3: Controlled Normal-Traffic Feature Shift**

Introduces a controlled distribution shift in normal traffic beginning with the fifth stream window. Feature values of normal traffic are shifted and scaled, while attack traffic remains unchanged.

The transformation uses a fixed random seed (42). The scale values are sampled from a uniform distribution between 1.15 and 1.45, while the shift values are sampled from a normal distribution with mean 0.65 and standard deviation 0.25.

This scenario evaluates how the replay methods respond to changes in normal-traffic feature distributions.

### Training Configuration

| Parameter | Value |
|---|---|
| Classifier | Multilayer Perceptron (MLP) |
| Hidden layers | 256, 128 |
| Activation | ReLU |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Epochs per window | 5 |
| Replay memory capacity | 5,000 |
| Attack pool allocation | 70% (3,500 samples) |
| Normal pool allocation | 30% (1,500 samples) |
| Main benchmark PSI threshold | 0.35 |
| Maximum normal-pool refresh | 35% |
| Random seeds | 42, 101, 202 |

## 5. DADR Memory Management

The replay memory has a fixed capacity of 5,000 samples.

### Attack Pool (70%)

The attack pool has a capacity of 3,500 samples. Its available capacity is distributed across the attack classes observed so far. When a class has more samples than its allocated quota, samples are randomly selected to maintain the allocation.

### Normal Pool (30%)

The normal pool has a capacity of 1,500 samples. Reservoir sampling is used to maintain a representative set of normal traffic during the initial filling process.

Once the pool is full, the PSI detector monitors incoming normal samples. When drift is detected, up to 35% of the normal-pool capacity can be replaced with incoming normal samples. The actual number of replacements is limited by the number of available incoming samples.

### PSI Drift Detection

The detector uses 10 quantile-based bins for each feature. Reference bin boundaries are obtained from stored normal samples, and incoming normal samples are evaluated using the same bins.

PSI is calculated for each feature, and the median feature-level PSI is used as the overall drift score.

The main benchmark uses a threshold of **0.35**. A drift event is triggered when the median PSI reaches or exceeds this value.

The detector requires at least 10 reference and 10 incoming normal samples. Features with insufficient unique reference-bin boundaries are not used to produce a non-zero PSI score.

The PSI calculation does not require attack labels. However, the complete replay-buffer update uses labels to separate normal and attack traffic.

## 6. Main Benchmark Results

The following results are reported as mean ± standard deviation across three random seeds.

### Scenario S1: Original Chronological Stream

| Method | Final Cum. Macro-F1 | Mean Cum. Macro-F1 | Attack Recall | False Alerts / 10k | Forgetting | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 0.1093 ± 0.0052 | 0.2460 ± 0.0026 | 0.2174 ± 0.0082 | 135.57 ± 16.46 | 0.5044 ± 0.0310 | 228.1 |
| EWC | 0.1668 ± 0.0014 | 0.2537 ± 0.0014 | 0.2248 ± 0.0100 | 156.99 ± 2.23 | 0.4410 ± 0.0249 | 320.5 |
| LwF | 0.1051 ± 0.0015 | 0.1789 ± 0.0105 | 0.1004 ± 0.0114 | 41.28 ± 7.31 | 0.3817 ± 0.0325 | 255.2 |
| ER-Reservoir | 0.3937 ± 0.0436 | 0.5322 ± 0.0179 | 0.4702 ± 0.0395 | 119.60 ± 3.48 | 0.3251 ± 0.0727 | 232.6 |
| ER-ClassBalanced | 0.6034 ± 0.0262 | 0.6643 ± 0.0100 | 0.7316 ± 0.0218 | 213.45 ± 17.17 | 0.1794 ± 0.0137 | 223.1 |
| Attack-Only | 0.5173 ± 0.0311 | 0.6325 ± 0.0175 | 0.7046 ± 0.0482 | 242.78 ± 3.54 | 0.1751 ± 0.0090 | 230.6 |
| DADR-NoDrift | 0.5610 ± 0.0359 | 0.6551 ± 0.0168 | 0.7184 ± 0.0264 | 200.98 ± 3.27 | 0.1802 ± 0.0143 | 232.6 |
| **DADR** | **0.5610 ± 0.0359** | **0.6551 ± 0.0168** | **0.7184 ± 0.0264** | **200.98 ± 3.27** | **0.1802 ± 0.0143** | **213.0** |
| Joint (Offline Reference) | 0.6050 ± 0.0064 | 0.7238 ± 0.0151 | 0.7167 ± 0.0332 | 124.10 ± 10.06 | 0.0672 ± 0.0408 | 1510.9 |

DADR and DADR-NoDrift produce the same evaluation metrics in S1 because no drift events were triggered in this scenario.

### Scenario S3: Controlled Normal-Traffic Feature Shift

| Method | Final Cum. Macro-F1 | Mean Cum. Macro-F1 | Attack Recall | False Alerts / 10k | Forgetting | Triggers | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive | 0.0824 ± 0.0331 | 0.2194 ± 0.0067 | 0.2363 ± 0.0297 | 1588.18 ± 245.32 | 0.5514 ± 0.0641 | 0 | 239.0 |
| EWC | 0.0807 ± 0.0286 | 0.2242 ± 0.0053 | 0.2395 ± 0.0105 | 1625.21 ± 200.02 | 0.4906 ± 0.0488 | 0 | 300.8 |
| LwF | 0.0952 ± 0.0304 | 0.1708 ± 0.0132 | 0.1408 ± 0.0141 | 1049.85 ± 87.79 | 0.4605 ± 0.0357 | 0 | 236.7 |
| ER-Reservoir | 0.4723 ± 0.0154 | 0.5678 ± 0.0273 | 0.5411 ± 0.0645 | **169.36 ± 14.50** | 0.2591 ± 0.0062 | 0 | 211.2 |
| ER-ClassBalanced | 0.3002 ± 0.0238 | 0.5042 ± 0.0104 | 0.8077 ± 0.0456 | 1688.47 ± 52.01 | 0.1311 ± 0.0113 | 0 | 202.1 |
| Attack-Only | 0.3197 ± 0.0016 | 0.4942 ± 0.0097 | **0.8136 ± 0.0500** | 1931.87 ± 70.94 | **0.1219 ± 0.0075** | 0 | 211.8 |
| DADR-NoDrift | **0.4324 ± 0.0201** | 0.5916 ± 0.0072 | 0.8017 ± 0.0423 | 467.93 ± 17.69 | 0.1289 ± 0.0191 | 0 | 212.8 |
| **DADR** | 0.4153 ± 0.0239 | **0.5940 ± 0.0063** | 0.7954 ± 0.0324 | 450.18 ± 17.88 | 0.1889 ± 0.0397 | **6** | 208.8 |
| Joint (Offline Reference) | 0.6431 ± 0.0281 | 0.7314 ± 0.0151 | 0.7245 ± 0.0284 | 72.98 ± 6.96 | 0.0384 ± 0.0374 | 0 | 1511.3 |

DADR achieves the highest mean cumulative Macro-F1 among the online methods in S3. Compared with DADR-NoDrift, it records slightly fewer false alerts, although its final cumulative Macro-F1 and attack recall are lower.

The Joint method is an offline reference that uses all available data and is not directly comparable to the online methods in terms of training conditions.

**Evaluation note:** Cumulative metrics are calculated after adaptation using subsets of the initial training data and previously observed stream windows. They measure post-adaptation performance on observed data and should not be interpreted as independent test performance. Prequential predictions are made before training on each incoming window and provide a more direct measure of performance on newly arriving traffic.

## 7. Visual Results

### Figure 1. Comparison Across S1 and S3

![S1 and S3 comparison](results/figures/fig1_s1_vs_s3_comparison.png)

### Figure 2. Cumulative Macro-F1 Across Stream Windows

![Window trajectories](results/figures/fig2_window_trajectories.png)

### Figure 3. Detection Performance and False-Alert Trade-off

![F1 versus false alerts](results/figures/fig3_f1_vs_false_alerts_tradeoff.png)

### Figure 4. Drift Detection and False Alerts in S3

![S3 drift and false alerts](results/figures/fig4_s3_drift_and_false_alerts.png)

## 8. Hyperparameter Ablation

Ablation experiments were conducted on S3 to examine the effect of replay memory capacity, attack-to-normal pool ratio, and PSI threshold.

### Memory Budget

| Capacity | Mean Cum. Macro-F1 | Final Cum. Macro-F1 | Attack Recall | False Alerts / 10k |
|---|---:|---:|---:|---:|
| 1,000 | 0.5368 ± 0.0137 | 0.3198 ± 0.0156 | 0.7355 ± 0.0337 | 819.48 ± 17.19 |
| 2,000 | 0.5589 ± 0.0046 | 0.3750 ± 0.0195 | 0.7694 ± 0.0506 | 663.00 ± 29.57 |
| **5,000** | **0.5940 ± 0.0063** | 0.4153 ± 0.0239 | 0.7954 ± 0.0324 | **450.18 ± 17.88** |
| 10,000 | 0.5931 ± 0.0164 | **0.4185 ± 0.0334** | **0.8084 ± 0.0463** | 492.01 ± 12.96 |

### Attack-to-Normal Pool Ratio

| Attack / Normal | Mean Cum. Macro-F1 | Final Cum. Macro-F1 | Attack Recall | False Alerts / 10k |
|---|---:|---:|---:|---:|
| 100% / 0% | 0.4942 ± 0.0097 | 0.3197 ± 0.0016 | 0.8136 ± 0.0500 | 1931.87 ± 70.94 |
| 80% / 20% | 0.5848 ± 0.0063 | 0.3941 ± 0.0317 | **0.8495 ± 0.0119** | 646.67 ± 16.24 |
| **70% / 30%** | **0.5940 ± 0.0063** | 0.4153 ± 0.0239 | 0.7954 ± 0.0324 | **450.18 ± 17.88** |
| 50% / 50% | 0.5832 ± 0.0119 | **0.4271 ± 0.0387** | 0.8034 ± 0.0593 | 484.24 ± 17.05 |

### PSI Threshold

| PSI Threshold | Mean Cum. Macro-F1 | Final Cum. Macro-F1 | Attack Recall | False Alerts / 10k | Triggers |
|---|---:|---:|---:|---:|---:|
| 0.10 | 0.5737 ± 0.0014 | 0.3829 ± 0.0266 | 0.8201 ± 0.0452 | 561.78 ± 19.55 | 8.00 |
| 0.15 | 0.5737 ± 0.0014 | 0.3829 ± 0.0266 | 0.8201 ± 0.0452 | 561.78 ± 19.55 | 8.00 |
| 0.20 | **0.5940 ± 0.0063** | **0.4153 ± 0.0239** | 0.7954 ± 0.0324 | **450.18 ± 17.88** | 6.00 |
| 0.25 | 0.5751 ± 0.0025 | 0.3891 ± 0.0035 | 0.8089 ± 0.0550 | 569.76 ± 9.65 | 6.67 |
| 0.30 | 0.5749 ± 0.0028 | 0.3919 ± 0.0064 | 0.7981 ± 0.0364 | 572.25 ± 12.04 | 6.00 |

The PSI threshold values in this table are separate ablation settings. The main benchmark configuration uses a threshold of **0.35**.

### Figure 5. Hyperparameter Ablation Results

![Hyperparameter ablation](results/figures/fig5_ablation_study.png)

## 9. Reproducing the Experiments

### Requirements

The project uses Python 3.10 or later and PyTorch 2.0 or later. The dependencies are listed in `requirements.txt`.

The original experiments were run in Google Colab using an NVIDIA Tesla T4 GPU with 16 GB of VRAM.

### Step 1. Clone the Repository

```bash
git clone https://github.com/ayesha-ml/DADR-Continual-IDS.git
cd DADR-Continual-IDS
```

### Step 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3. Prepare the Dataset

Download the official CIC-IDS2017 MachineLearningCSV dataset from:

https://www.unb.ca/cic/datasets/ids-2017.html

Place the eight CSV files in:

```text
raw_data/cicids2017/
```

The dataset directory should match the path configured in `src/config.py`.

### Step 4. Run the Main Benchmark

```bash
python -m scripts.run_main_benchmark
```

This runs the main benchmark across the two scenarios, nine methods, and three random seeds.

### Step 5. Generate Figures

```bash
python -m scripts.plot_results
```

The experiment results and generated figures are saved in the repository's results directory.

**Note:** Exact numerical results may vary across hardware, library versions, and sources of randomness. The reported results are based on the experimental configuration and seeds listed above.

## 10. Limitations

The current experiments have several limitations:

- Evaluation is limited to CIC-IDS2017.
- S3 uses a controlled shift in normal-traffic features and does not represent every type of real network drift.
- The complete replay-buffer update requires class labels to separate normal and attack traffic.
- Cumulative metrics include previously observed data and should not be treated as independent test results.
- The experiments use three random seeds, and further evaluation on additional datasets and drift conditions is needed.

These limitations provide directions for future work, including testing on additional intrusion detection datasets, evaluating more types of concept drift, and investigating settings with delayed or unavailable labels.

## 11. Acknowledgements

The experiments use the CIC-IDS2017 dataset provided by the Canadian Institute for Cybersecurity at the University of New Brunswick.

