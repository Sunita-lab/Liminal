# Liminal

## 1. Core Idea
Liminal is an experimental system to analyze how public attention distributes across domains, 
and whether that attention aligns with structural indicators of real-world impact.

It studies patterns of amplification, neglect, and temporal spikes 
using lightweight machine learning and designed impact metrics.

## 2. Dataset
- 200 headlines
- 2 domains: environment & entertainment
- Time window: 15 Feb 2026 – 1 Mar 2026
- 33 sources

## 3. Week 1 Work

### Observational Layer
- Daily volume
- Volatility & burstiness
- Vocabulary comparison
- Sentiment analysis

### Classification Layer
- Basic feature model (~58% accuracy)
- TF-IDF + Logistic Regression (~82% accuracy)

### Structural Weight
- PTI (Physical Threat Index)

## 4. Key Observations
- Entertainment headlines show higher volatility (std ≈ 9.75) than environment (std ≈ 3.39).
- Entertainment volume spikes concentrated around event-driven days (Feb 27 – Mar 1).
- Environment headlines show more question-based framing (9% vs 5%).
- TF-IDF model significantly outperformed basic feature model (82% vs 58% accuracy).
- PTI scores were consistently higher for environment domain.
- Attention volume and PTI do not appear proportionally aligned.

## 5. Next Direction
Divergence Layer (Attention vs Impact)