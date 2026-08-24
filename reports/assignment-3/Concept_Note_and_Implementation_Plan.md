# Concept Note and Implementation Plan

## Project Information

**Project Title:** Modelling and Improving Brand Visibility in AI Assistant Recommendations

**Team Members:** Kübra Gezici / Furkan Karlı / Murat Mert Küçük / Zeynep Sinal

**Repository:** `edasaruhan/SIC_AI_17_Capstone_Group_8`

---

# Concept Note

## 1. Project Overview

Product discovery is moving from search engines to conversational AI assistants. A search engine returned ten ranked links and left selection to the user; an assistant returns two or three brand names and the decision is effectively made. There is no second page, no listing below the fold, and no paid placement inside the organic answer. A brand that is not named is not ranked lower — it is absent from the consideration set entirely.

**Marketing context.**

| Dimension | This project |
|---|---|
| Target customer (system user) | Digital marketing and GEO agencies serving SME and mid-market clients; secondarily PR and corporate communication agencies |
| End beneficiary | Small, local and challenger brands without the equity of global incumbents |
| Product / service | A visibility diagnostic tool delivered as a web interface and an agency-ready report |
| Channel analysed | Conversational AI assistants, with and without web retrieval |
| Customer journey stage | Awareness and consideration — specifically shortlist formation |
| Business process | Content and earned-media planning; marketing budget allocation |

**Customer and business impact.** Agencies currently face a question they cannot answer: a client asks why they never appear in ChatGPT or Gemini recommendations, and the agency has no diagnostic instrument. Existing tools count mentions retrospectively but cannot explain them. The consequence is that content budget is spent on intuition.

Our system changes the decision rather than the dashboard. A brand invisible because the model does not recognise it needs sustained brand investment measured in quarters. A brand invisible because the content written about it is weak can be fixed in weeks at a fraction of the cost. These are opposite budget decisions, and no current tool separates them. For an agency the measurable value is a new billable deliverable and a defensible allocation recommendation; for the brand it is avoided waste and faster recovery of a discovery channel it is currently losing.

## 2. Objectives and Marketing KPIs

**Objectives.** (1) Measure brand visibility in AI assistant recommendations for Turkish queries, where no measurement currently exists. (2) Build a model that predicts which brand will be recommended from the content the assistant retrieves. (3) Separate the contribution of brand recognition from the contribution of content. (4) Convert that separation into prioritised, verifiable content actions.

**Marketing KPIs the project influences.**

| KPI | Definition | Baseline | Target / success criterion |
|---|---|---|---|
| **AI Share of Voice (ASoV)** | % of category responses mentioning the brand | Unknown for Turkish — established by this project | Baseline table published for 30 brands across 3 sectors |
| **Top-1 recommendation rate** | % of responses where the brand is the primary pick | Measured in Layer A | Model predicts it above naive baseline |
| **Content-attributable visibility share** | Portion of visibility explained by content rather than recognition | No existing measurement | Reported per sector with 95% confidence interval |
| **Prescription actionability** | Agency judgement of whether recommended actions are executable | None | ≥ 2 of 3 agencies rate the report actionable in validation |

**Technical success criteria (proxies for the above).**

| Criterion | Baseline | Target |
|---|---|---|
| Top-1 accuracy | Naive rule: pick brand of first-ranked source | Beat by ≥ 10 percentage points |
| PR-AUC (row level) | Class prevalence ≈ 0.125 | ≥ 0.60 |
| Prior-only model (M0) | Brand recognition alone | Full model beats it materially, with clustered bootstrap CIs excluding overlap |
| Calibration | — | Brier score reported; probabilities calibrated before client-facing use |

The project is judged successful if the Turkish baseline is published, the model beats both naive and prior-only baselines with reported confidence intervals, and the attribution is produced with an interpretable prescription that agencies confirm they could act on.

## 3. Background and Marketing Context

**Why this matters now.** Assistant-mediated discovery has become a primary research channel, and organic traffic to brand properties is declining as a result. Teams that invested heavily in SEO have no equivalent instrument for the new channel and, in many cases, do not know whether their content is cited, summarised or ignored.

**Who is affected.** Most acutely, small and local brands. Research shows US-centric models favour globally known brands over local equivalents, and that when products are otherwise indistinguishable the known brand is recommended almost always. The disadvantage is structural, not a result of product quality.

**Current practice and its limits.** A commercial category has formed rapidly — Profound, AthenaHQ, Peec AI, Otterly, Scrunch, Adobe LLM Optimizer, plus extensions from Semrush and Ahrefs. Their architecture is a scheduled prompt runner plus mention counting, reported as ASoV over time. Three limitations follow: they *count* rather than *predict*; they do not separate recognition from content; and they are English-centric and priced for enterprise buyers, leaving Turkish queries and SME-focused agencies unserved. In Turkey, several agencies now market GEO services, but as service providers using these same foreign tools — no local measurement capability exists.

**Why AI is the right approach.** The problem is intrinsically an AI problem in two senses. The channel being measured is an AI system, so the only way to observe it is to query it systematically and treat its outputs as data. And the signal that determines recommendation lives in subtle properties of natural language — specificity versus vagueness, hedging, authority markers, evidential quality — which cannot be captured by keyword counting. This requires NLP: a supervised model over retrieved text, combined with interpretable feature-based modelling to make the output actionable.

## 4. Proposed AI Methodology

**Task formulation.** Given a query and the set of candidate brands with the web content supporting each, predict which brand the assistant recommends. Trained as binary classification over (response, candidate brand) pairs and evaluated listwise, since the real decision is a selection among candidates.

**Model family.** Models are built cumulatively, because the marketing insight lies in the *differences* between them, not in any single score.

| Model | Input | Algorithm | Purpose |
|---|---|---|---|
| Naive baselines | — | Rules (first-ranked source's brand; most frequent brand) | Floor that must be beaten |
| M0 | Brand prior only | Logistic regression | Isolates recognition effect |
| M1 | + position, source type | Logistic regression | Structural effects |
| M2 | + engineered language features | LightGBM + SHAP | **Production model** — produces prescriptions |
| M3 | Retrieved snippet text | Cross-encoder (BERTurk for Turkish, XLM-R multilingual) | Highest expected accuracy; enables ablation |
| M3b | Sentence embeddings + M2 features | LightGBM | Fallback and control |

**Key methodological device — masking ablation.** M3 is trained twice on identical data: once with brand names visible in the snippets, once replaced by a `[BRAND]` token. The performance gap isolates how much of the decision is name recognition versus content. Combined with including or excluding the brand-prior feature, this yields a three-condition attribution table. This is the mechanism that produces our differentiating KPI and it is why a text-based model is required — masking is a text-level intervention that a feature-only model cannot perform.

**Why this approach fits.** Interpretability is a functional requirement, not a preference: the deliverable is "change this, expect roughly this much," so SHAP-compatible gradient boosting is the production choice. The transformer earns its place through the ablation rather than accuracy alone. Data scale (thousands of grouped examples) dictates base-size models; larger models would overfit. Cost is negligible — approximately $8–40 for the entire data generation — so model choice is driven by quality and provider diversity rather than budget. We explicitly reject prompting an LLM to judge visibility directly: it is circular, unreproducible, and cannot explain *why*.

**Evaluation.** Technical: PR-AUC as primary (class prevalence ≈ 12.5% makes accuracy misleading), ROC-AUC secondary, response-level top-1 accuracy and NDCG@3 as business-facing metrics. Confidence intervals via clustered bootstrap at query level, because 20–30 repetitions per cell break independence. Calibration assessed with Brier score. Generalisation tested by leaving out one sector, one model and one language in turn. Marketing: agency validation, in which the report is produced for a real brand and reviewed for actionability; and consistency of prescriptions with the controlled Layer B interventional evidence.

## 5. Architecture / Workflow Design Diagram

![Solution architecture](figures/fig-architecture.png)

**Component roles.**

1. **Query and brand design** — defines what is measured: Turkish queries across three sectors, a brand universe spanning global, local, small and validated fictional brands, and the controlled variant grid for the interventional layer.
2. **AI assistant APIs** — the measurement instrument. Three providers, retrieval on/off conditions, candidate lists of eight brands per call, randomised ordering, 20–30 repetitions.
3. **Raw capture** — every call stored as JSONL with response text, retrieved search results, model version and timestamp; `run_id` allows resuming after interruption. Raw data is never modified.
4. **Reference dataset** — the open English dataset provides the schema, validates our method, and supplies the cross-language comparison.
5. **Parsing and feature preparation** — converts responses into (response, candidate brand) pairs, normalises brand names, tags source domains, extracts language features, computes the brand prior from the retrieval-off condition, and freezes the query-level grouped split.
6. **Model family and evaluation** — as described in Section 4.
7. **Visibility report** — the marketing output: ASoV against category average, the recognition-versus-content attribution, and ranked actions with estimated gains.
8. **Ethical filter** — blocks fabricated-authority recommendations before they reach the user.
9. **Marketing action** — the report becomes an agency content brief, earned-media target list and budget allocation.
10. **KPI measurement and human review** — visibility is re-measured after action, forming the feedback loop; an agency strategist reviews all recommendations before execution.

## 6. Data Sources

The project uses one open reference dataset, one dataset we generate, and one secondary benchmark. The reference dataset is `3RAIN/brand-bias-evaluations` (HuggingFace, MIT licence): 9,586 assistant responses across four models and four domains, with 10 queries per domain, roughly 30 repetitions per cell, and two conditions — with and without web retrieval. Each record contains the response text, the queries the model issued, the full organic search results retrieved, and structured labels extracted by a judge model, including `top_recommendation`, `brand_mentions` with position, and `confidence_in_extraction`; the format is nested JSON, flattened to Parquet. Its decisive property is that it pairs *what the model saw* with *what it recommended*, which is what makes supervised learning of the decision possible, while the retrieval-off condition provides a direct empirical measure of each brand's prior recognition. We generate a parallel Turkish dataset on the same schema: Layer A (observational — 3 sectors × 12 queries × 2 conditions × 3 models × 20–30 repetitions, approximately 4,300 calls yielding around 34,000 labelled rows through the candidate-list design) and Layer B (interventional — 30 brands × 8 controlled description variants × 2 models × 10 repetitions, approximately 4,800 calls). E-GEO (arXiv:2511.20867) serves as a secondary English e-commerce benchmark for generalisation testing. Major preprocessing steps are: use only the `all` subset of the reference data to avoid double counting; parse nested JSON fields; normalise brand names with a curated alias dictionary; tag result domains as affiliate, editorial or official through a rule-based list built by two independent annotators with agreement reported; extract language features from versioned lexicons; compute brand priors from the retrieval-off condition; exclude low-confidence extraction rows from primary training; and freeze a query-level grouped split, since repeated sampling makes rows non-independent and random splitting would leak. **Responsible use:** no personal, customer or user data is collected or processed at any stage — inputs are synthetic personas and public brand information, and outputs are model-generated text — so no consent or anonymisation obligations arise; collected responses remain subject to the providers' terms of use, which is documented in the published data card.

## 7. Literature and Industry Review

Academic work converges on four propositions. LLM recommendations exhibit systematic bias toward incumbent and globally known brands, exceeding the popularity bias of classical recommender systems and disadvantaging local brands specifically (Lichtenberg et al., 2024; Kamruzzaman et al., 2024). This dominance is nonetheless conditional: Chu & Hou (2026) show that when products are specified identically the known brand wins essentially always, but the advantage collapses once any differentiating quality signal appears, with product parameters explaining the large majority of ranking variance and brand identity only a marginal share — meaning the real barrier for challengers is absence of differentiating information rather than brand equity itself. Content is therefore a genuine lever: Aggarwal et al. (2024) formalise Generative Engine Optimization and report roughly 40% visibility gains from content rewriting alone, Filandrianos et al. (2025) show that credibility-oriented language (authority, social proof) shifts recommendations substantially while overt promotional language does not, and Pfrommer et al. (2024) demonstrate that list position independently affects rankings — a methodological constraint that makes order randomisation mandatory. Industry practice has moved faster than academia: platforms including Profound and Adobe's LLM Optimizer have standardised ASoV as the operational metric, Adobe reports multi-fold citation increases and a 41% rise in LLM referral traffic after applying GEO discipline internally, and agency case studies report citation rates rising from single digits to the mid-twenties within a quarter — though these are vendor-published, self-reported, and lack control conditions or independent verification. **The gap this capstone addresses** is therefore fourfold: the entire literature is English with Turkish unmeasured; every study is descriptive, producing rates rather than a reusable predictive model; no work separates recognition-driven from content-driven visibility, which is precisely the distinction that determines budget allocation; and the industry evidence has no open reproducible baseline against which its claims can be assessed.

---

# Implementation Plan

## 1. Technology Stack

**Language and environment.** Python 3.11; `uv` for dependency management; configuration in YAML with `pydantic-settings`; secrets in `.env`, never committed; fixed random seeds throughout.

**Data collection.** `httpx` with `asyncio` for concurrent requests; `tenacity` for retry with exponential backoff; provider SDKs (`anthropic`, `openai`, `google-generativeai`); `pydantic` for response schema validation; JSONL via `jsonlines` for raw capture; `rich` and `loguru` for progress and logging. Model providers: three distinct providers selected for diversity, using budget and mid-tier models (Gemini Flash-Lite tier, Claude Haiku, GPT budget tier) with batch API where latency permits.

**Data processing and storage.** `pandas` for the main pipeline, `duckdb` for SQL over JSONL and Parquet, `pyarrow`/Parquet as the derived format, `datasets` for loading the reference dataset. Versioning through `data/raw`, `data/interim`, `data/processed` plus a committed `splits.json`.

**Text processing and features.** `regex` and `unicodedata` for Turkish normalisation (particularly İ/ı handling); `rapidfuzz` for brand-name matching; `tldextract` for domain parsing; `textstat` for readability; `zeyrek` for Turkish morphological analysis where lexicon matching requires it. Language lexicons stored as versioned YAML under `lexicons/`.

**Modelling.** `scikit-learn` and `lightgbm` for M0–M2; `torch`, `transformers` and `accelerate` for M3, with `dbmdz/bert-base-turkish-cased` (BERTurk), `xlm-roberta-base` and `microsoft/deberta-v3-base`; `sentence-transformers` for M3b; `optuna` for hyperparameter search; `mlflow` (local) for experiment tracking; `shap` and `captum` for explainability; `scipy` and `statsmodels` for statistics. Training on Google Colab or Kaggle free-tier GPU with checkpointing.

**Interface and delivery.** `streamlit` for the report interface; `plotly` and `matplotlib` for visualisation; deployment to Streamlit Community Cloud or HuggingFace Spaces.

**Quality and collaboration.** Git and GitHub with short-lived task branches and pull requests; `ruff`, `black`, `pre-commit`; `pytest` for parsing, normalisation and split-integrity tests; `make` targets for `setup`, `format`, `check`; GitHub Actions running `make check` on every pull request.

## 2. Timeline and Task Distribution

![Implementation timeline](figures/fig-gantt.png)

**Task distribution matrix.** Responsibilities continue the Sprint 0 allocation. R = responsible, S = supporting.

| Work package | Kübra | Furkan | Murat | Zeynep |
|---|---|---|---|---|
| Collection infrastructure, API orchestration | S | **R** | S | |
| Query pool, brand universe, variant grid | S | | S | **R** |
| Layer A and Layer B execution and monitoring | | **R** | S | |
| Parsing, brand normalisation, grouped split | | S | **R** | |
| Affiliate domain tagging (two annotators) | | | **R** | **R** |
| Language feature engineering and lexicons | **R** | | S | |
| Baselines, M0–M2, SHAP | **R** | | S | |
| M3 cross-encoder, M3b, masking ablation | **R** | S | | |
| Statistics, calibration, generalisation tests | **R** | | S | |
| Concentration and fairness analyses | S | | **R** | |
| Streamlit interface and ethical filter | S | **R** | | S |
| Agency validation round | | | | **R** |
| Dataset publication and data card | | S | **R** | |
| Technical report, demo, presentation | S | S | S | **R** |

Sprint 2 carries the highest load; collection runs in parallel with annotation and feature work. Working rhythm: 30-minute planning on Monday, 30-minute review on Friday, one summary note per sprint committed to the repository.

## 3. Milestones and Deliverables

| # | Milestone | Target date | Evidence of completion |
|---|---|---|---|
| M1 | Turkish data collection complete | 29 Aug | Raw JSONL in place; parse-success and cost report; collection log |
| M2 | Data ready for modelling | 3 Sep | `pairs.parquet`, frozen `splits.json`, leakage test passing, data card draft |
| M3 | Baseline established | 4 Sep | Naive and M0 scores recorded; comparison table committed |
| M4 | Interpretable model working | 7 Sep | M2 trained, SHAP figures produced, first prescriptions generated |
| M5 | Turkish visibility baseline published | 7 Sep | ASoV table for 30 brands across 3 sectors; concentration and fairness analyses |
| M6 | Attribution validated | 13 Sep | Three-condition ablation table with clustered bootstrap confidence intervals |
| M7 | Working prototype | 16 Sep | Streamlit app producing a full report from brand + sector input; ethical filter unit-tested |
| M8 | User test completed | 16 Sep | Reports produced for ≥ 2 agencies; written feedback recorded |
| M9 | Dataset published | 17 Sep | Turkish dataset and data card publicly available with licence |
| M10 | Final evaluation and submission | 18 Sep | Technical report, demo, presentation; pipeline reproducible from clean clone |

## 4. Challenges and Mitigation Strategies

| Challenge | Consequence if unmanaged | Mitigation | Fallback |
|---|---|---|---|
| **Data leakage from random splitting** | Inflated scores; invalid claims to clients | Query-level grouped split frozen at M2, protected by automated test | — (non-negotiable) |
| Collection slips past 29 Aug | All downstream modelling delayed with no slack | Daily monitoring; resume via `run_id`; hard checkpoint at 28 Aug | Halve Layer B variants; drop third model |
| API rate limits or cost overrun | Incomplete dataset | Candidate-list design (8× efficiency); batch API; daily spend cap; pilot-measured limits | Reduce to two providers; reduce repetitions to 20 |
| Small and imbalanced dataset | Transformer overfits | Base-size models only; three seeds; grouped CV; PR-AUC not accuracy | M3b embedding hybrid; report M2 as production model |
| **Deep learning fails to beat gradient boosting** | "Why deep learning?" challenge | Reported honestly — at our data scale this is a legitimate finding | M2 ships as production model; M3 retained solely for the ablation |
| Model memorises brand names | Content effect unmeasurable | The masking ablation measures this directly and reports it | Report as a finding rather than a failure |
| Turkish parsing failures | Silent data loss | Strict output templates; parse tests; per-model success rates reported | Replace underperforming model after pilot |
| Judge-model label noise | Unreliable ground truth | Exclude low-confidence rows; 100-row manual verification with agreement reported | Restrict analysis to high-confidence subset |
| Provider model updates mid-project | Learned relationships go stale | All collection within a two-week window; version and timestamp on every row | Stability re-run quantifies drift and is reported as a finding |
| Agency validation does not materialise | No marketing-side evidence | 15 organisations contacted to secure 3 sessions; scheduled early | Internal expert review against a written rubric |
| **Difficulty proving true marketing impact** | Claims exceed evidence | Layer B provides controlled interventional evidence for a feature subset; elsewhere estimates are labelled as estimates | Frame outputs as expected values with confidence, never guarantees |

## 5. Ethical and Responsible AI Considerations

**Privacy and personal data.** The project collects no personal, customer or user data at any stage. Inputs are synthetic personas and publicly available brand information; outputs are model-generated text. No consent, anonymisation or data-subject obligations arise. Provider terms of use governing collected responses are documented in the published data card. Interview notes from agency conversations are stored without personally identifying detail and are not committed to the repository.

**Manipulation risk — the central ethical issue.** The literature demonstrates that fabricated authority claims — invented clinical citations, non-existent expert endorsements — are among the most effective levers for increasing visibility. A system that optimised visibility naively would surface fabrication as its top recommendation, making the tool an instrument for deceiving consumers. Our mitigation is structural rather than advisory: the recommendation layer is constrained at code level to verifiable-evidence and source-diversity strategies, a blocked-template list is maintained, and an output filter is applied and unit-tested. This constraint is a deliberate product decision, and we accept that it caps measurable performance.

**Fairness and bias.** The project's subject matter *is* algorithmic bias. We report a model-by-model fairness comparison showing how each assistant treats local and small brands relative to global incumbents, and a concentration measure quantifying how winner-takes-all each category is. We also acknowledge bias in our own instrument: our brand universe, sector selection and Turkish lexicons embed choices that shape results, so all are published for scrutiny.

**Transparency and explainability.** Every prediction is accompanied by an attribution, never a bare score. The method, dataset, lexicons and code are published openly — a deliberate contrast with commercial tools whose methodology is proprietary and unverifiable.

**Avoiding misleading outputs.** Three safeguards: probabilities are calibrated before being shown, so a stated figure means what it claims; estimated gains are labelled as expected values under current model behaviour, not guarantees; and the surrogate-model limitation is stated in the interface itself, not only in documentation.

**Human oversight.** The intended user is an agency strategist, not an automated pipeline. Output is a prioritised recommendation set with confidence indicators, requiring human judgement to convert into a content plan. No output is published or executed automatically.

**AI assistance disclosure.** Generative AI tools were used during the preparation of this document for drafting, structuring, literature synthesis and figure generation, and during the project for code assistance. All content has been reviewed, verified and edited by the team; the research design, methodological decisions and conclusions are our own, and all cited sources have been checked against their originals.

## 6. References

Adobe (2026). *How to improve brand visibility in AI search engines.* Adobe Business Blog. https://business.adobe.com/blog/improve-brand-visibility-in-ai-search-engines

Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K. & Deshpande, A. (2024). *GEO: Generative Engine Optimization.* Proceedings of KDD 2024, pp. 50–61.

Bagga, P. S., Wu, Y., Aggarwal, P. & Deshpande, A. (2025). *E-GEO: A Testbed for Generative Engine Optimization in E-Commerce.* arXiv:2511.20867.

Chu, X. & Hou, Y. (2026). *Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems.* arXiv:2606.17443.

Filandrianos, G., Dimitriou, A., Lymperaiou, M., Thomas, K. & Stamou, G. (2025). *Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations.* Proceedings of EMNLP 2025.

Kamruzzaman, M., Nguyen, H. M. & Kim, G. L. (2024). *"Global is good, local is bad?": Understanding Brand Bias in LLMs.* Proceedings of EMNLP 2024, pp. 12704–12721.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30.

Kumar, A. & Lakkaraju, H. (2024). *Manipulating Large Language Models to Increase Product Visibility.* arXiv:2404.07981.

Lichtenberg, J. M., Buchholz, A. & Schwöbel, P. (2024). *Large Language Models as Recommender Systems: A Study of Popularity Bias.* arXiv:2406.01285.

Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.

Pfrommer, S., Bai, Y., Gautam, T. & Sojoudi, S. (2024). *Ranking Manipulation for Conversational Search Engines.* Proceedings of EMNLP 2024, pp. 9520–9534.

Schweter, S. (2020). *BERTurk — BERT models for Turkish.* Zenodo. Model: `dbmdz/bert-base-turkish-cased`

Sundararajan, M., Taly, A. & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* Proceedings of ICML 2017.

Three Rivers AI Nexus (2026). *What Brands Does Your AI Prefer?* Dataset: `3RAIN/brand-bias-evaluations`, HuggingFace, MIT License.

Documentation: HuggingFace `datasets`, `transformers`; LightGBM; SHAP; Streamlit; Anthropic, OpenAI and Google Gemini API references (accessed August 2026).
