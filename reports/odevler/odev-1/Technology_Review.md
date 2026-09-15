# Technology Review

## AI Technologies for Predicting and Diagnosing Brand Visibility in AI Assistants

**AI in Marketing Capstone** · Technology Review Submission

---

## 1. Introduction

### 1.1 What is being reviewed

This review evaluates the AI technologies available to build a system that predicts which brand an AI assistant will recommend, attributes that prediction between brand recognition and content quality, and converts the attribution into content actions a marketing team can execute.

Four technology families are in scope:

1. **Large language model APIs** — used as the measurement instrument that generates our data
2. **Gradient-boosted decision trees** — interpretable modelling over engineered features
3. **Transformer encoders (fine-tuned)** — modelling directly from retrieved text
4. **Explainability methods** — converting model output into marketing prescriptions

Also reviewed are the **commercial AI-visibility platforms** that represent the current state of practice.

### 1.2 Why a technology review matters here

The obvious technical answer is not the right marketing answer. The highest-accuracy model would be a large fine-tuned transformer, but a marketing deliverable is not a probability — it is *"do this, expect roughly this much improvement."* That requirement makes interpretability a functional specification rather than a nice-to-have, and it changes which technology wins.

A second reason is that this problem has an unusual structure: we are training a model to predict another model's behaviour. That makes our system a **surrogate model**, with consequences for validity, drift and how confidently results can be communicated to a client. Those consequences need to be understood before, not after, the technology is chosen.

---

## 2. Technology Overview

### 2.1 LLM APIs as a measurement instrument

Commercial assistant APIs (Anthropic Claude, OpenAI GPT, Google Gemini families) are used here not as the product but as the **object of study** and the data-generation mechanism. Relevant capabilities: web-retrieval tool use, which lets us compare recommendations with and without external content; temperature control, enabling repeated sampling to estimate stable rates; and structured output instructions, which make responses machine-parsable at scale.

Current rates span roughly $0.10–$5.00 per million input tokens and $0.40–$30.00 per million output tokens across tiers. Batch processing offers approximately 50% discounts, and prompt caching up to 90% on repeated input. For our workload (~11,000 calls, ~12.7M tokens), total cost lands between $2 and $40 — economically insignificant, which means model choice can be driven by quality and diversity rather than budget.

In marketing more broadly, these APIs power content generation, audience research synthesis, conversational commerce and customer-service automation. Our application is unusual in that we treat the model as the *channel to be measured* rather than the tool doing the work.

### 2.2 Gradient-boosted decision trees

LightGBM and XGBoost are the standard approach for tabular prediction. They handle mixed feature types, non-linear interactions and class imbalance well, train in seconds on datasets of our size, and are the workhorse behind most production marketing models — churn prediction, propensity scoring, lead scoring and uplift modelling.

Their decisive property for this project is compatibility with SHAP, which decomposes a single prediction into per-feature contributions. That decomposition is literally our product's prescription layer.

### 2.3 Fine-tuned transformer encoders

Encoder models (BERT family) fine-tuned for classification or ranking capture semantic and pragmatic properties of text that hand-engineered features cannot: specificity versus vagueness, hedging, authority framing, evidential quality. For our task the appropriate architecture is a **cross-encoder** that jointly encodes the query and the snippets supporting each candidate brand, trained listwise to select the recommended brand.

Language coverage matters. For Turkish, BERTurk (`dbmdz/bert-base-turkish-cased`) is the established option; XLM-RoBERTa provides multilingual comparison; DeBERTa-v3 is the strongest English base model. Given our data scale (thousands, not millions, of grouped examples) base-size models are appropriate; larger models would overfit.

In marketing, this family underpins sentiment analysis, intent classification, review mining and content moderation.

### 2.4 Explainability methods

SHAP for tree models and integrated gradients for transformers convert predictions into attributions. In most marketing ML deployments explainability is a compliance or trust feature. Here it is the product: without attribution, the system outputs a score no one can act on.

### 2.5 Commercial AI-visibility platforms

Profound, AthenaHQ, Peec AI, Otterly, Scrunch, Adobe LLM Optimizer and extensions from Semrush and Ahrefs constitute a crowded and well-funded category. Their common architecture is a scheduled prompt runner plus mention-counting analytics, reported as AI Share of Voice over time.

---

## 3. Relevance to the Marketing Project

### 3.1 Where each technology sits in the marketing workflow

| Marketing workflow stage | Technology | Output |
|---|---|---|
| **Channel measurement** — establish baseline visibility | LLM APIs, repeated sampling | ASoV, top-1 rate, mean rank by brand and category |
| **Diagnosis** — why is the brand absent? | Transformer + masking ablation; retrieval on/off contrast | Content-attributable visibility share |
| **Prescription** — what should change? | GBDT + SHAP | Ranked content actions with estimated gains |
| **Planning** — where to spend | Attribution split | Content budget vs brand-building budget decision |
| **Measurement loop** — did it work? | Re-run measurement | Before/after ASoV comparison |

### 3.2 Supporting the KPIs

The technology choices map directly to the KPI set. ASoV and top-1 rate require only repeated API sampling with disciplined controls. Mean rank requires the ranking protocol. The novel KPI — content-attributable visibility share — requires the masking ablation, which requires a text-based model; it cannot be produced by a feature-based model alone, because masking a brand name is a text-level intervention. This single requirement is why the project needs a transformer at all.

### 3.3 The decision the technology supports

The commercial output is not a dashboard but a budget recommendation. A brand at 12% ASoV against a category average of 31% receives an attribution: if the shortfall is predominantly content-driven, the recommended action is content and earned-media investment measured in weeks; if it is predominantly recognition-driven, the recommendation is sustained brand investment measured in quarters, and content spend would largely be wasted. No current tool makes this distinction.

---

## 4. Comparison and Evaluation

### 4.1 Evaluation criteria

Criteria are weighted for the marketing task rather than for benchmark performance:

| Criterion | Why it matters here |
|---|---|
| Predictive quality | Must beat naive baselines to be credible |
| **Interpretability** | The prescription *is* the product |
| Data requirements | We have thousands of grouped rows, not millions |
| Training cost and time | Six-week schedule, free-tier GPU |
| Turkish capability | Core differentiator |
| Maintainability | Model drift requires periodic retraining |
| Deployment simplicity | Four-day final sprint |

### 4.2 Model family comparison

| Approach | Predictive quality | Interpretability | Data need | Cost | Turkish | Verdict |
|---|---|---|---|---|---|---|
| Naive baselines (first-ranked source; most frequent brand) | Low | Total | None | None | N/A | **Required floor** — if not beaten, nothing is learned |
| Logistic regression on prior only (M0) | Low | Total | Minimal | Negligible | N/A | **Control** — isolates brand-recognition baseline |
| LightGBM on engineered features (M2) | Medium-high | **High (SHAP)** | Low | Negligible | Feature-level | **Primary production model** |
| Fine-tuned cross-encoder (M3) | **Highest expected** | Medium (IG) | Medium | Moderate (GPU) | BERTurk / XLM-R | **Required for masking ablation** |
| Sentence embeddings + LightGBM (M3b) | Medium-high | Medium | Low | Low | Multilingual | **Fallback and control** |
| Prompting an LLM to judge directly | Unknown, unstable | None | None | Per-call | Native | **Rejected** — circular and non-reproducible |

### 4.3 Rationale for the chosen architecture

We adopt a **cumulative family** rather than a single model, because the marketing question requires the *differences between models*, not just the best score. M0 quantifies what brand recognition alone explains; M2 adds structural and linguistic features and provides the prescriptions; M3 adds raw text and enables the masking ablation. The value of the project lies in the gaps between these models.

The rejected option deserves comment: prompting an LLM to score visibility directly would be fastest but is circular — using a model to judge model behaviour without ground truth — and cannot produce stable, reproducible attributions. It fails the marketing requirement precisely because it cannot say *why*.

### 4.4 Measurement instrument selection

For data generation we select three models from three different providers. Provider diversity is not redundancy: the fairness comparison across assistants is one of the project's outputs, and it is meaningless within a single provider. Budget-tier models are selected where parse reliability permits, with per-model parse-success rates measured in the pilot and any unreliable model replaced.

### 4.5 Evaluation metrics

Standard accuracy is inappropriate given ~12.5% positive class. We use PR-AUC as primary, ROC-AUC as secondary, and response-level top-1 accuracy and NDCG@3 as the business-facing metrics — top-1 accuracy being the one an agency intuitively understands. Confidence intervals use clustered bootstrap at query level, following Chu & Hou (2026), because repeated sampling breaks independence. Calibration is assessed with Brier score, since a probability presented to a client must mean what it says.

---

## 5. Use Cases and Examples

### 5.1 Adobe — GEO applied to its own properties

Adobe reports that applying GEO discipline to Adobe.com produced a fivefold increase in citations for one product, a 200% increase in LLM visibility, and a 41% increase in LLM referral traffic for another product area, within weeks.

**What this demonstrates:** the marketing problem is real at enterprise scale, the lever is content structure and clarity, and the outcome is measurable in familiar terms including referral traffic. **What we learn:** visibility responds fast to content changes, which supports our product's premise that content-attributable shortfalls are correctable in weeks rather than quarters.

### 5.2 Agency-published GEO case studies

Published cases report a B2B SaaS moving from single-digit to mid-twenties percentage citation rates over 90 days with attributed pipeline value; a fintech tripling AI Share of Voice in eight weeks; a healthcare platform going from no citations to appearing in a majority of relevant answers. Typical interventions are structured data markup, long-form content answering specific high-intent prompts, and comparison pages.

**What this demonstrates:** practitioners have converged on ASoV as the operational KPI and treat AI visibility as a managed growth channel. **What we learn:** our outputs must be denominated in ASoV to be adopted, and the intervention types reported (structured content, third-party comparison coverage) should appear in our prescription vocabulary.

**Critical caveat.** These are vendor- and agency-published results, self-reported, without control conditions or independent verification, and with obvious selection bias toward successes. They establish what the market believes and buys; they do not establish reliable effect sizes. This absence of a reproducible baseline is itself part of our project's justification.

### 5.3 Academic demonstration

Aggarwal et al. (2024) provide the controlled counterpart, reporting visibility improvements of roughly 40% from content modification under experimental conditions. Filandrianos et al. (2025) narrow this further by showing which language families are responsible — credibility signals rather than promotional ones.

**What we learn:** the academic evidence corroborates the direction of the industry claims while providing the methodological rigour they lack — which is the combination our project aims to reproduce in Turkish.

### 5.4 The commercial platform category

Profound and comparable platforms have raised substantial funding and serve enterprise clients at monthly price points that place them out of reach for the SME-focused agencies we target. Their architecture is a prompt scheduler plus mention counter.

**What we learn:** the measurement layer is commoditised and we should not compete there. The unoccupied position is diagnosis and prediction, at a price point and in a language the incumbents do not serve.

---

## 6. Limitations, Risks and Opportunities

### 6.1 Limitations

**Surrogate-model validity.** We model an assistant's behaviour, not ground truth about product quality. Predicted gains are expected values under current model behaviour, not guarantees. This must be stated plainly in client-facing output.

**Temporal drift.** Provider model updates invalidate learned relationships. Mitigated by recording version and timestamp on every row, completing collection within a two-week window, and quantifying drift with a stability re-run — which converts a limitation into a reported finding.

**Category scope.** The model is trained on specific sectors; a brand outside them cannot be scored directly. Positioning must be "a method proven in three categories and retrainable," not "a universal tool."

**Third-party platform dependency.** The entire measurement layer depends on commercial APIs whose pricing, rate limits and retrieval behaviour we do not control.

**Explainability limits.** SHAP attributions are associational, not causal. Layer B's controlled variants provide genuine interventional evidence for a subset of features; where they do not, prescriptions are framed as estimates.

### 6.2 Risks

| Risk | Marketing consequence | Mitigation |
|---|---|---|
| Data leakage via random splitting | Inflated scores, invalid client claims | Query-level grouped splits, frozen and leakage-tested |
| Model memorises brand names | Content effect unmeasurable | Masking ablation measures this directly and reports it honestly |
| **Brand safety / dual use** | Tool could recommend fabricated authority claims | Blocked-template list, output filter, unit-tested; verifiable-evidence strategies only |
| Hallucinated or unparsable responses | Silent data loss | Strict output format, parse-success monitoring per cell |
| Over-automation | Agency applies prescriptions without judgement | Output framed as prioritised recommendations with confidence, not instructions |
| Deep learning fails to beat GBDT | "Why deep learning?" challenge | Reported honestly; at our data scale this is a legitimate and informative finding |

### 6.3 Opportunities

**Human oversight by design.** The intended user is an agency strategist, not an automated pipeline. Presenting attribution and confidence alongside every recommendation keeps the human in the decision.

**Open baseline.** Publishing the Turkish dataset and the method creates the independently verifiable reference the category currently lacks — a differentiator against closed commercial tooling.

**Integration.** Because output is a ranked content brief, it slots directly into an agency's existing content workflow rather than requiring a new one.

**Extension.** The same pipeline retrains for new sectors or languages; the measurement design is the reusable asset, not the specific model weights.

---

## 7. Conclusion

The technology decision is a **cumulative family with LightGBM as the production model and a fine-tuned cross-encoder as the analytical instrument**, with LLM APIs serving as the measurement layer and SHAP plus integrated gradients as the prescription layer.

This choice is driven by the marketing requirement rather than benchmark performance. A marketing deliverable must say why and what to change, so interpretability functions as a specification; LightGBM with SHAP satisfies it and trains in seconds on our data scale. The transformer earns its place not through accuracy alone but because the masking ablation — the mechanism producing our differentiating KPI — is only possible with a text-based model. Direct LLM prompting is rejected as circular and unreproducible despite being fastest.

If implemented successfully, the practical value is a decision aid that converts an unmeasured, intuition-driven budget question into an evidenced one: an agency can tell a client not only that they are absent from AI recommendations, but what share of that absence is correctable through content, which specific actions to take, and roughly what improvement to expect. In the Turkish market, where no measurement currently exists at all, this moves the state of practice from anecdote to measurement.

---

## 8. References

Adobe (2026). *How to improve brand visibility in AI search engines.* Adobe Business Blog. https://business.adobe.com/blog/improve-brand-visibility-in-ai-search-engines

Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K. & Deshpande, A. (2024). *GEO: Generative Engine Optimization.* Proceedings of KDD 2024, pp. 50–61.

Chu, X. & Hou, Y. (2026). *Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems.* arXiv:2606.17443.

Filandrianos, G., Dimitriou, A., Lymperaiou, M., Thomas, K. & Stamou, G. (2025). *Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations.* Proceedings of EMNLP 2025.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems 30.

Kumar, A. & Lakkaraju, H. (2024). *Manipulating Large Language Models to Increase Product Visibility.* arXiv:2404.07981.

Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems 30.

Schweter, S. (2020). *BERTurk — BERT models for Turkish.* Zenodo. Model: `dbmdz/bert-base-turkish-cased`

Sundararajan, M., Taly, A. & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* Proceedings of ICML 2017.

Three Rivers AI Nexus (2026). *What Brands Does Your AI Prefer?* Dataset: `3RAIN/brand-bias-evaluations`, HuggingFace, MIT License.

*Pricing figures* reflect published provider rates as of 16 August 2026 and are subject to change; see the project's cost model (`maliyet_modeli.xlsx`) for the current calculation.
