# Liminal — Attention vs Impact Analysis

## Core Question

Do attention patterns reflect real-world impact, or distort it?

---

## Overview

Liminal analyzes how public attention distributes across domains, and whether it aligns with structural indicators of real-world impact.

The study compares two domains:

- **Environment** — higher real-world consequences  
- **Entertainment** — higher public attention  

---

## Dataset

- 200 headlines  
- 2 domains (environment, entertainment)  
- Time window: Feb 15 – Mar 1, 2026  
- 33 sources  

---

## Approach

The analysis was structured across three layers:

### Observational Layer
- Daily volume  
- Volatility (standard deviation)  
- Burstiness (attention spikes)  
- Sentiment patterns  

### Representation Layer
- TF-IDF + Logistic Regression  
- Compared with basic feature model  

### Impact Layer
- Physical Threat Index (PTI) as a proxy for real-world impact  

---

## Key Results

- **Volatility:**  
  Entertainment shows higher variability (~9.75 vs ~3.39)

- **Burstiness:**  
  ~2× higher in entertainment (~1.07 vs ~0.50)

- **Model Performance:**  
  TF-IDF model outperformed basic features (82% vs 58%)

- **Impact (PTI):**  
  Environment carries higher real-world threat signals  

- **Mismatch:**  
  Attention volume does not align proportionally with impact  

---

## Insight

Attention is not proportional to impact.

High-impact domains remain relatively stable,  
while lower-impact domains dominate attention spikes.

---

## Structure

- `notebooks/liminal_analysis.ipynb` — main analysis  

---

## Status

Ongoing exploratory project.

Current results indicate a measurable attention–impact gap,  
but the system is still evolving — including dataset expansion,  
refinement of impact metrics, and deeper validation.

