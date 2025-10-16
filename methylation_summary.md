# DNA Methylation: Key Concepts and Insights

## What is DNA Methylation?

- DNA methylation is the addition of a **methyl group (–CH₃)** to DNA, usually at **CpG sites** (cytosine followed by guanine).
- It does not alter the DNA sequence, but it changes how the DNA is read and interpreted.
- Functions like a **switchboard for gene activity**.

---

## Illumina Methylation Arrays

Illumina arrays (450K, EPIC/850K) measure methylation at CpG sites with two signals:

- **M (methylated signal)**
- **U (unmethylated signal)**

From these, two main metrics are derived: **M-values** and **Beta values**.

---

## M-value

- Formula:  
  $$ M = \log_2 \left( \frac{Methylated + \alpha}{Unmethylated + \alpha} \right) $$
- Interpretation:
  - **Positive M** → more methylated signal (hypermethylated)
  - **Negative M** → more unmethylated signal (hypomethylated)
  - **~0** → about 50% methylated
- Used for **statistical modeling** due to better variance stabilization.

---

## Beta Value

- Formula:  
  $$ \beta = \frac{M}{M + U + \alpha} $$
- Range: 0–1 (interpretable as percentage methylation)
  - **0** → unmethylated
  - **1** → fully methylated
  - **0.5** → ~50% methylation
- Intuitive and often used for **visualization and interpretation**.

---

## Converting Between M-value and Beta

- $ M = \log_2\left( \frac{\beta}{1 - \beta} \right) $
- $ \beta = \frac{2^M}{2^M + 1} $

Python functions:

```python
def beta_to_m(beta, alpha=1e-6):
    beta = np.clip(beta, alpha, 1 - alpha)
    return np.log2(beta / (1 - beta))

def m_to_beta(m):
    return 2**m / (2**m + 1)
```

---

## Why Methylation is Interesting

- **Gene regulation:** High methylation in promoters → gene silencing; low methylation → gene activation.
- **Development:** Guides cell differentiation (e.g., liver vs brain).
- **Aging:** Methylation patterns change predictably → epigenetic clocks.
- **Cancer:** Aberrant methylation silences tumor suppressors or activates oncogenes.
- **Clinical applications:**
  - Biomarkers (e.g., blood-based methylation tests for cancer detection)
  - Therapies (e.g., demethylating agents in leukemia)
  - Personalized medicine (predicting treatment response).

---

## Summary

- **M-values** = log2 ratios, best for statistics.  
- **Beta values** = proportions (0–1), best for interpretation.  
- Methylation acts as an **epigenetic switch** controlling gene activity, with roles in development, aging, and disease.  
- It holds strong promise in diagnostics, prognostics, and therapeutics.
