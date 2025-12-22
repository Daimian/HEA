# Data-driven design of mechanically hard soft(-magnetic) high-entropy alloys (HEAs)

<img width="600" height="600" alt="Figure2" src="https://github.com/user-attachments/assets/584b3d9e-3a4c-41bc-a66e-50c85fd93ec1" />

This repository contains data, scripts, and examples for **data-driven discovery and optimization of high-entropy alloys (HEAs)** that aim to combine:
- **mechanical hardness proxies** (e.g., bulk modulus, elastic descriptors), and
- **soft-magnetic proxies** (e.g., high saturation magnetization / magnetic moment and Curie temperature).

It consolidates two tightly connected components:
1. **High-throughput EMTO–CPA database + ML surrogate modeling** (Advanced Science, 2025).
2. **Multi-objective Bayesian optimization (MOBO) loop** for accelerated composition search (arXiv:2509.05702).

---

## Related publications

- **Advanced Science (2025)**: *Data-Driven Design of Mechanically Hard Soft Magnetic High-Entropy Alloys*  
  DOI: 10.1002/advs.202500867  
  Full text (open access / PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12097130/

- **arXiv (2025)**: *Accelerated Design of Mechanically Hard Magnetically Soft High-entropy Alloys via Multi-objective Bayesian Optimization*  
  arXiv:2509.05702  
  https://arxiv.org/abs/2509.05702

If you use this repository, please cite the papers above (see [Citation](#citation)).

---

## Project highlights

### 1) High-throughput database (HTP-DFT / EMTO–CPA)
- Large-scale DFT dataset (EMTO–CPA) spanning **equimolar quaternary and quinary HEAs**.
- For each composition, multiple **magnetic states** and **crystal prototypes** are evaluated:
  - magnetic: **FM vs PM**
  - structure: **BCC vs FCC**
- Target properties include:
  - **B0**: bulk modulus (EOS-fit)
  - **mag / M_tot**: total magnetic moment (proxy for saturation magnetization)
  - **TC**: Curie temperature (mean-field estimate in the database workflow)

### 2) Ensemble ML surrogate modeling
- Train predictive models to map (composition + structure/volume descriptors) → (phase/structure + target properties).
- Use ensemble learning to improve robustness and generalization in a large and diverse chemical space.

### 3) MOBO extension (accelerated search)
The MOBO workflow (arXiv:2509.05702) demonstrates:
- a **10-element pool** (Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn),
- exploration of **equimolar and non-equimolar quinary** compositions,
- multi-objective optimization over:
  - **Pugh’s ratio** (B/G) and **Cauchy pressure** (C12 − C44) as mechanical descriptors,
  - **Total magnetization**,
  - **Curie temperature (TC)**,
- with an ensemble surrogate (bootstrapping + stacking) and Monte-Carlo-based acquisition evaluation.

---
