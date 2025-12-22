# MOBO module — Accelerated design of mechanically hard & magnetically soft HEAs

This repository contains the **multi-objective Bayesian optimization (MOBO)** component used in our workflow for discovering **mechanically hard yet magnetically soft high-entropy alloys (HEAs)**. The MOBO loop couples:
1) candidate generation in a high-dimensional composition space,
2) first-principles evaluation (EMTO-CPA + magnetic/elastic post-processing),
3) descriptor-based surrogate modeling (heterogeneous ensemble),
4) Monte-Carlo-based multi-objective acquisition,
to iteratively identify a **Pareto-optimal set** under conflicting magnetic and mechanical targets.  
(See the workflow schematic and the MOBO description in the manuscript.) 

---

## 1. Optimization problem

### 1.1 Design space
- **Element pool (10 elements):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn.
- **Compositions:** not restricted to fixed element subsets; the algorithm explores **arbitrary quinary** (five-element) systems with **non-equimolar ratios** in the 10-element simplex via Monte Carlo sampling. 
- **Crystal prototypes:** BCC and FCC are considered as candidate lattices in the evaluation stage (and the final Pareto candidates reported in the manuscript stabilize in BCC). 

### 1.2 Objectives (4 targets)
We optimize four objectives simultaneously:
- **Total magnetic moment** \(M_{\mathrm{Tot}}\) 
- **Curie temperature** \(T_C\) 
- **Pugh’s ratio** \(B/G\) 
- **Cauchy pressure** \(C_{12}-C_{44}\)

These are tracked during the MOBO iterations and jointly define the Pareto front. 

---

## 2. Surrogate model (heterogeneous ensemble with uncertainty)

Instead of a single Gaussian process surrogate, we use a **heterogeneous ensemble** that is more stable in sparse-data, high-dimensional descriptor spaces.

### 2.1 Base learners
The ensemble aggregates multiple algorithm families (examples in the manuscript include neural networks, LightGBM, CatBoost, Random Forests, ExtraTrees, kNN). 

### 2.2 Bootstrapping + OOB scoring
For each base algorithm \(M\), we:
- draw \(B\) bootstrap resamples,
- train submodels \(M,b\),
- compute **out-of-bag (OOB) \(R^2\)** scores to quantify reliability. 

### 2.3 Stacking + adaptive weighting
Predictions are further combined by stacking, and the final algorithm weight \(\hat{\omega}_M\) blends:
- an OOB-based score \(w'_M\),
- a stacking-based score \(w''_M\),
using a tunable factor \(\alpha\in[0,1]\).  
The ensemble outputs:
- predictive mean \(\mu(x)\),
- predictive uncertainty \(\sigma(x)\),
which are used by the acquisition step. 

---

## 3. Acquisition + Monte Carlo candidate proposal

### 3.1 Batch BO protocol
- Start with **10 random** initial compositions.
- At each iteration, propose and evaluate a **batch of 10** new compositions. 

### 3.2 Multi-objective acquisition
- The acquisition is based on **expected hypervolume improvement (EHVI)**.
- EHVI (and related quantities) are estimated **numerically via repeated sampling** from the ensemble predictive distribution, rather than relying on closed-form expressions. 

### 3.3 Stopping criterion
The loop terminates when the **hypervolume (HV) gain** falls below **0.1%** for **10 consecutive iterations**. In practice, the manuscript reports no appreciable gain beyond ~15 iterations under the chosen configuration. 

### 3.4 Pseudocode
A compact version of the workflow is given as “Algorithm 1: Meta-adaptive hybrid Bayesian optimization with Monte Carlo sampling” in the manuscript. :contentReference[oaicite:11]{index=11}

---

## 4. Outputs: Pareto set and top candidates (Table 1)

After convergence, the algorithm returns the **non-dominated (Pareto) set** discovered during the search, together with its estimated Pareto front. 

### 4.1 What Table 1 contains
**Table 1** reports the **top 20 candidates** on the four-objective Pareto front, selected by **hypervolume contribution**. Columns are:
- `Formula` : composition string (element fractions sum to 1)
- `MTot [µB/site]`
- `TC [K]`
- `B/G` [dimensionless]
- `C12 − C44 [GPa]` 

### 4.2 Key patterns in Table 1 (as described in the manuscript)
- **All top 20 candidates stabilize in BCC.**
- Highest-ranked alloys are typically **Fe–Co rich**, often with **Mn and Ni**, yielding large \(M_{\mathrm{Tot}}\) and \(T_C\).
- Candidates with stronger mechanical scores (higher \(B/G\), higher \(C_{12}-C_{44}\)) tend to include **significant Cu**.
- **Zn and Ti** appear only in small amounts among the top candidates; the search progressively de-emphasizes Zn, Ti, V. 

### 4.3 Property ranges across the top 20 (Table 1)
Across the reported top-20 Pareto candidates:
- \(M_{\mathrm{Tot}}\): **0.54 – 0.71** \(\mu_B/\)site  
- \(T_C\): **1160 – 1737** K (several exceed **1600 K**)  
- \(B/G\): **2.41 – 2.97**  
- \(C_{12}-C_{44}\): **36 – 45** GPa 

---
