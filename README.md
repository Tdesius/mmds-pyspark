# Mining Massive Data Sets — End-Term Project

Three large-scale data mining tasks implemented with **Apache PySpark** for distributed computation: agglomerative hierarchical clustering in non-Euclidean space, CUR-compressed linear regression for gold price prediction, and LSH-accelerated collaborative filtering.

> 504048 – Mining Massive Data Sets · Ton Duc Thang University  
> Advised by Nguyen Than An · Academic Year 2025–2026

📄 Full report: [report.pdf](./report/report.pdf) (IEEE conference format)

---

## Tasks Overview

| Task | Problem | Key Technique | Result |
|---|---|---|---|
| 1 | String clustering | Agglomerative + Jaccard + PySpark RDD | 800 → 743 clusters, stops at step 57 |
| 2 | Vietnamese gold price prediction | CUR dimensionality reduction + PySpark LinearRegression | R² ≈ 1.0, Test RMSE 0.46 at k=15 |
| 3 | Movie recommendation | Collaborative Filtering + Random-hyperplane LSH | 12–16× speed-up, <2% RMSE penalty |

---

## Task 1 — Hierarchical Clustering in Non-Euclidean Spaces

**Data:** 10,000 randomly generated lowercase alphabetical strings (length 32–64), tokenized into 4-shingles. Clustering performed on a representative 800-string subset.

**Algorithm:** Bottom-up agglomerative clustering (Approach 2 for non-Euclidean spaces):
- **Centroid:** most-central element (minimizes average Jaccard distance to cluster members)
- **Inter-cluster distance:** Jaccard distance between cluster representatives
- **Merge threshold:** 80th percentile of initial pairwise distances
- **Termination:** diameter-jump criterion — halts when `global_avg_dist` increases by factor γ = 1.5

**Results:**

| Step | global_avg_dist | Clusters |
|---|---|---|
| 0 | 0.000 | 800 |
| 20 | 0.314 | 780 |
| 40 | 0.668 | 760 |
| 56 | 0.911 | 744 |
| **57 (stop)** | **1.351** | **743** |

Jump factor at step 57: ×1.48 > γ = 1.5 → algorithm halts. 3D t-SNE projection (perplexity=30, 1000 iterations) confirms visually distinct cluster regions.

---

## Task 2 — Gold Price Prediction with CUR Dimensionality Reduction

**Data:** 5,563 daily Vietnamese SJC gold buy prices (2009-08-01 – 2025-01-01). Sliding 15-day lag window → 5,548 supervised samples, split 70/30.

**Algorithm:** `CURReducer` class uses PySpark `RowMatrix` SVD to compute leverage scores and select the k most informative lag features. 11 experiments: k ∈ {15, 14, …, 5}.

**Results:**

| k (features) | Train RMSE | Test RMSE |
|---|---|---|
| 15 (100%) | 0.42 | 0.46 |
| 10 (67%) | 0.48 | 0.52 |
| 5 (33%) | 0.64 | 0.71 |

Reducing features by **67%** raises Test RMSE by only **0.25M VND/tael** — less than 1% relative error against typical gold prices of >80M VND/tael. CUR correctly identifies the most informative lag features.

---

## Task 3 — Collaborative Filtering with LSH

**Data:** `ratings2k.csv` (2,365 ratings) for primary experiments; `ratings20k.csv` (19,911 ratings) for scalability profiling. 70/30 train/test split.

**Algorithm:** User–user CF with mean-centred cosine similarity (handles sparse, scale-heterogeneous vectors). Random-hyperplane LSH (K=8 bits, L=10 tables) for O(1) average candidate retrieval.

**Accuracy (ratings2k.csv):**

| N (neighbours) | RMSE (LSH) | RMSE (Brute) | Δ RMSE |
|---|---|---|---|
| 5 | 0.961 | 0.938 | +0.023 |
| 10 | 0.924 | 0.912 | +0.012 |
| 20 | 0.907 | 0.899 | +0.008 |
| 30 | 0.898 | 0.893 | +0.005 |

**Speed-up:**

| N | LSH (ms) | Brute (ms) | Speed-up |
|---|---|---|---|
| 5 | 0.8 | 12.4 | **15.5×** |
| 30 | 1.1 | 13.1 | **11.9×** |

Scalability test on ratings20k.csv: LSH latency stays <2ms as user count grows from 200 to 2,000. Brute-force reaches ~120ms at 2,000 users.

---

## Project Structure

```
├── task1.ipynb                  # Hierarchical clustering
├── task02.ipynb                 # Gold price prediction + CUR
├── task3.ipynb                  # Collaborative filtering + LSH
├── report/
│   └── report.pdf               # IEEE conference format report
├── data/
│   ├── dataset.csv              # Generated string corpus (task 1)
│   ├── gold_prices.csv          # SJC gold prices 2009–2025 (task 2)
│   ├── ratings2k.csv            # Ratings dataset (task 3)
│   └── ratings20k.csv           # Extended ratings for scalability
└── README.md
```

---

## Setup & Usage

```bash
pip install pyspark numpy pandas matplotlib scikit-learn
jupyter notebook
```

Each notebook is self-contained. PySpark requires Java 8+ installed and `JAVA_HOME` set.

> Google Colab users: notebooks install dependencies via `!pip install` cells.

---

## Individual Contributions

| Member | Task |
|---|---|
| Pham Quoc Hung | Task 1 · Report |
| Nguyen Nam Phong, Huynh Huu Minh | Task 2 |
| Dinh Bui Khanh Huy | Task 3 · Report |

*Ton Duc Thang University — Faculty of Information Technology*


---

## License

This project (source code) is released under the [MIT License](LICENSE).

## Data Credits

- `gold_prices.csv` — Vietnamese SJC gold price data provided by the course instructor. Not redistributed in this repository.
- `ratings2k.csv` — Rating dataset provided by the course instructor. Not redistributed in this repository.
- `ratings20k.csv` — Synthetically generated dataset for scalability testing.
- `dataset.csv` — Generated programmatically by `task1.ipynb` (random string corpus).
