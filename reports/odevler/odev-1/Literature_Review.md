# Literature Review

## Brand Visibility in AI-Assistant-Mediated Product Discovery

**AI in Marketing Capstone** · Literature Review Submission
Project: Modelling and Improving Brand Visibility in AI Assistant Recommendations

---

## 1. Introduction

### 1.1 The marketing problem

Product discovery is migrating from search engines to conversational AI assistants. A search engine returned ten ranked links and left selection to the user; a generative assistant returns two or three brand names and the decision process ends there. In this channel there is no second page, no organic listing beneath the fold, and — in most current deployments — no paid placement inside the organic answer. A brand that is not named is not merely ranked lower; it is absent from the consideration set entirely.

For marketers this is a discovery-channel disruption with three consequences. First, the shortlist is formed by the model rather than the user, so classical funnel assumptions about browsing and comparison no longer hold. Second, the selection criteria are undocumented, so brands cannot diagnose why they are absent. Third, existing measurement infrastructure (rank tracking, click-through, impression share) does not apply, because there are no impressions to count.

### 1.2 Marketing context

| Dimension | This project |
|---|---|
| **Target customer (user of our system)** | Digital marketing and GEO agencies serving SME and mid-market clients; secondarily PR and corporate communication agencies |
| **End beneficiary** | Small, local and challenger brands that lack the brand equity of global incumbents |
| **Channel** | Conversational AI assistants (with and without web retrieval) |
| **Customer journey stage** | Awareness and consideration — specifically shortlist formation, the moment the candidate set is defined |
| **Marketing outcome to improve** | Presence and prominence of a brand inside AI-generated recommendations |
| **Business decision supported** | Where to allocate marketing budget: brand-building investment versus content and earned-media investment |

### 1.3 Measurable objectives (KPIs)

The literature has not converged on a standard metric, so we define the KPI set this project measures and optimises:

- **AI Share of Voice (ASoV)** — the proportion of responses in a category in which the brand is mentioned. This is the closest equivalent to impression share in the new channel and is emerging as the de facto industry North Star metric.
- **Top-1 recommendation rate** — the proportion of responses in which the brand is the primary recommendation. Commercially the most consequential, because assistants typically surface very few names.
- **Mean rank** — average position when mentioned, capturing prominence rather than mere presence.
- **Content-attributable visibility share (proposed)** — the portion of a brand's visibility explained by the content written about it rather than by prior brand recognition. This is a diagnostic KPI we introduce; no existing tool or paper reports it, and it is the metric that converts measurement into a budget decision.

### 1.4 Research questions

- **RQ1.** Can the brand an assistant will recommend be predicted from the retrieved web content it sees?
- **RQ2.** What share of that decision is attributable to prior brand recognition versus the language and provenance of the content?
- **RQ3.** Does commercially motivated content (affiliate sources) measurably shift the recommendation?
- **RQ4.** Do these patterns generalise across models, languages and sectors — in particular, does Turkish behave like English?

### 1.5 Why a literature review is necessary

Three reasons. The academic literature already establishes *that* brand bias exists and *that* content can move recommendations, so replicating those findings would waste project time. The industry literature has moved faster than the academic literature and defines the metrics practitioners actually buy, which our system must speak to. And the methodological choices in this field are unforgiving: prior work documents position bias, non-independence of repeated sampling, and the fragility of fictional-brand controls — all failure modes we must design against before collecting data.

---

## 2. Organization

The literature is organised thematically, in the order a marketer encounters the problem: first *is the channel biased?*, then *can we influence it?*, then *how far can influence be pushed before it becomes manipulation?*, and finally *how do we measure any of it?*

- **Theme A — Brand and popularity bias as a structural property of the channel**
- **Theme B — Content and language as controllable marketing levers (GEO)**
- **Theme C — Ranking mechanics, manipulation and adversarial risk**
- **Theme D — Measurement infrastructure: benchmarks, datasets and industry metrics**

---

## 3. Summary and Synthesis

### 3.1 Theme A — Brand and popularity bias

**Lichtenberg, Buchholz & Schwöbel (2024)** examine whether LLMs used as recommenders inherit popularity bias. Working in the movie domain with standard recommendation benchmarks, they find popularity bias exceeding that of classical collaborative filtering. *Marketing contribution:* establishes that the long-tail disadvantage familiar from recommender systems is not merely reproduced but amplified in the LLM channel.

**Kamruzzaman, Nguyen & Kim (2024)** test global versus local brand preference across categories including footwear and clothing, using prompt-based elicitation on US-centric models. They find systematic favouring of global brands over local equivalents. *Marketing contribution:* the first direct evidence that market-entry disadvantage for local brands is encoded in the channel itself — the foundation of our Turkish-language research question (RQ4).

**Kumar & Lakkaraju (2024)** work in the coffee-machine category, manipulating product descriptions to test whether visibility can be shifted. Established products dominate even against equally specified alternatives, but targeted description changes move rankings. *Marketing contribution:* incumbency is real but not absolute — a controllable lever exists.

**Chu & Hou (2026)** provide the most complete treatment. Using product sets of one real and nine validated fictional brands across three commercial models, two languages, and 20–30 repetitions per cell, they describe a *Conditional Monopoly*: when products are specified identically the known brand is recommended essentially always, but this dominance collapses once any differentiating quality signal appears. Their variance decomposition attributes the large majority of ranking variance to product parameters, a small share to list position, and only a marginal share to brand identity — brand acts as a tiebreaker under information scarcity. *Marketing contribution:* reframes the barrier for challenger brands. The obstacle is not brand equity but the **absence of differentiating information** — a problem marketing can actually solve.

### 3.2 Theme B — Content and language as levers

**Aggarwal et al. (2024)** formalise Generative Engine Optimization as a task and introduce the GEO-Bench evaluation set. By systematically rewriting source content and measuring visibility inside generated answers, they report visibility improvements of roughly 40% from content modification alone. *Marketing contribution:* establishes GEO as a discipline and demonstrates that visibility is an addressable marketing variable rather than a fixed property.

**Filandrianos et al. (2025)** test five cognitive-bias language strategies — authority, social proof, scarcity, anchoring, loss aversion — at varying intensities. They find a two-tier pattern: credibility-oriented language (authority, social proof) shifts recommendations substantially, while overt sales language (scarcity, anchoring, loss aversion) has weak effect. *Marketing contribution:* directly actionable copywriting guidance, and a taxonomy that becomes our linguistic feature set.

**Bagga et al. (2025)** extend GEO to e-commerce with a testbed of over 7,000 realistic product queries and fifteen rewriting heuristics. *Marketing contribution:* moves GEO from web content to product listings, the setting closest to commercial practice.

**Lin et al. (2025)** show that even synonym-level substitution can shift brand recommendation probability considerably. *Marketing contribution:* the sensitivity of the channel to small wording changes implies that content quality control matters at a granularity most marketing teams do not currently operate at.

### 3.3 Theme C — Ranking mechanics and manipulation

**Pfrommer et al. (2024)** decompose ranking variance in conversational search engines into product name, description content and list position, showing position alone carries meaningful weight. *Marketing contribution and methodological consequence:* any measurement that does not randomise candidate order is measuring position, not content. This directly constrains our experimental design.

**Nestaas, Debenedetti & Tramèr (2025)** study adversarial prompt injection for search visibility and find prisoner's-dilemma dynamics among attackers. **Chu & Hou (2026)** replicate this incentive structure using ordinary commercial marketing language rather than adversarial injection, finding that when all brands optimise, individual advantage decays towards zero while non-participants receive effectively no recommendations. *Marketing contribution:* GEO is a competitive necessity rather than an advantage — opting out is costly, but universal adoption erodes returns. This is a strategic finding marketing leadership needs.

The manipulation literature also carries an ethical warning that is central to our project design: **Chu & Hou (2026)** report that fabricated authority claims — invented clinical citations and expert endorsements — are among the most effective visibility levers. A system that optimises visibility naively would recommend exactly this behaviour.

### 3.4 Theme D — Measurement infrastructure

**Three Rivers AI Nexus (2026)** publish an open evaluation dataset of 9,586 completed model responses spanning four models and four domains. The design attempted up to 30 repetitions per cell; 14 of 9,600 planned calls were not completed. Critically, the dataset stores the organic search results retrieved alongside each response. The publisher reports that NordVPN's VPN top-recommendation rate rose from 5.2% to 33.4% under the search-enabled condition, while affiliate-related warnings fell from 41.2% to 15.4%. *Marketing contribution:* the first open resource linking what a model *saw* to what it *recommended*, making the commercial content supply chain observable. Because the search-enabled condition also changes the system prompt, the contrast is strong observational evidence but not a fully isolated causal estimate of retrieval alone.

**Industry practice** has converged faster than academia. Platforms including Profound, AthenaHQ, Peec AI, Otterly and Adobe's LLM Optimizer track brand mentions across assistants, and agency-published case studies report outcomes such as multi-fold growth in AI Share of Voice within weeks and citation-rate increases from single-digit to mid-twenties percentages over a quarter. Adobe reports substantial increases in citations and LLM-referred traffic after applying GEO discipline to its own properties. *Caveat:* these are vendor- and agency-published results without independent verification or control conditions; they establish that practitioners measure ASoV and believe it movable, not that the reported magnitudes are reliable.

### 3.5 Comparative synthesis

| Study | Domain | Data | Method | Marketing metric | Key limitation |
|---|---|---|---|---|---|
| Lichtenberg et al. (2024) | Movies | Benchmark sets | Prompted recommendation | Popularity skew | Not a commercial brand setting |
| Kamruzzaman et al. (2024) | Apparel, footwear | Prompt elicitation | Comparative prompting | Global vs local mention rate | English only; no content lever |
| Kumar & Lakkaraju (2024) | Appliances | Synthetic listings | Description manipulation | Rank shift | Single category |
| Chu & Hou (2026) | Skincare (+ robustness) | 1 real + 9 fictional brands | Factorial, 3 models, 2 languages, 20–30 reps | Recommendation rate, rank | Fictional brands; descriptive only; preprint |
| Aggarwal et al. (2024) | Web content | GEO-Bench | Content rewriting | Visibility in answers | Not product/brand competition |
| Filandrianos et al. (2025) | Multi-category | Product descriptions | Bias-language injection | Recommendation shift | English only |
| Bagga et al. (2025) | E-commerce | 7,000+ queries | Rewriting heuristics | Visibility | English, Amazon-centric |
| Pfrommer et al. (2024) | Electronics | Conversational engines | Variance decomposition | Rank variance | Methodological, not applied |
| Three Rivers (2026) | VPN, travel, hosting, editors | 9,586 responses | Observational, search on/off | Mention and first-place rate | Not consumer goods; narrow brand universe; no model |

Read together, the literature converges on four propositions:

1. **The channel is structurally biased toward incumbents**, and this bias is strongest precisely when brands are otherwise undifferentiated (Chu & Hou; Kamruzzaman; Lichtenberg).
2. **Content is a genuine and measurable lever** — visibility responds to how a brand is described, not only to what the brand is (Aggarwal; Filandrianos; Kumar & Lakkaraju; Lin).
3. **Credibility signals dominate persuasion signals.** Authority and third-party validation move the channel; conventional promotional language largely does not (Filandrianos; Chu & Hou).
4. **The competitive equilibrium is unfavourable.** Early movers gain; universal adoption dissipates the gain; non-participation is penalised (Nestaas; Chu & Hou).

But the literature is uniformly **descriptive**: it reports rates, effect sizes and significance tests. No study produces a *predictive* artefact — a model that, given a brand's current content, estimates its probability of being recommended.

---

## 4. Marketing Relevance and Gap

### 4.1 What existing approaches do well

Academic work has credibly established the existence, direction and rough magnitude of brand bias, and has identified which language families move recommendations. Industry tooling has operationalised measurement: agencies can now track whether a client is mentioned, across which assistants, over time. Between them, a marketer can answer "am I visible?" and "is visibility movable in principle?"

### 4.2 What remains unresolved

**Gap 1 — Diagnosis, not just measurement.** Every commercial tool counts mentions retrospectively. None estimates the probability of recommendation for given content, and none separates the contribution of prior brand recognition from the contribution of content. This matters commercially because the two conditions demand opposite budget responses: a recognition deficit requires sustained brand investment over months, while a content deficit is correctable in weeks at far lower cost. Marketers currently make this allocation blind.

**Gap 2 — Language and market coverage.** The entire literature is English, partially Chinese. Turkish — and by extension most non-Anglophone markets — is unmeasured. Given Kamruzzaman et al.'s finding that models favour global over local brands, the local-market case is precisely where the effect is expected to be strongest and is precisely where no evidence exists.

**Gap 3 — Provenance of the content supply.** The Three Rivers dataset suggests that when retrieval is enabled, a large share of top-ranked sources are commercially motivated, and that the model's own caveating behaviour declines correspondingly. No study models the effect of source type on the recommendation decision.

**Gap 4 — Evaluation discipline in industry evidence.** Reported case-study gains lack control conditions and independent verification. There is no open, reproducible baseline against which agency claims can be assessed.

### 4.3 How this project responds

The project reframes the problem from measurement to **prediction and attribution**, and applies it in an unmeasured market:

1. **A predictive model** that estimates which brand an assistant will recommend from the retrieved content it sees, trained on open data and evaluated with grouped splits and clustered confidence intervals (addresses Gap 1, RQ1).
2. **A masking ablation** that trains the same model with brand names visible and masked, isolating recognition-driven visibility from content-driven visibility and yielding the *content-attributable visibility share* KPI (Gap 1, RQ2).
3. **A Turkish dataset** built on the schema of the English reference dataset, enabling the first local-market measurement and a cross-language comparison (Gap 2, RQ4).
4. **Source-type modelling** that treats affiliate, editorial and official provenance as features and quantifies their effect (Gap 3, RQ3).
5. **Open, reproducible method and data**, providing the independent baseline the industry evidence currently lacks (Gap 4).

The project also inherits an explicit ethical constraint from the literature: because fabricated authority claims are demonstrably effective, our recommendation layer is restricted to verifiable-evidence and source-diversity strategies, and must not surface fabrication as a tactic.

---

## 5. Conclusion

The literature supports four design decisions for this capstone.

First, **target the information gap, not brand equity.** Chu & Hou's finding that brand acts as a tiebreaker under information scarcity means the actionable marketing intervention is supplying differentiating, credible information — which is why our product output is a content prescription rather than a brand-awareness recommendation.

Second, **prioritise credibility features.** Filandrianos et al.'s two-tier result determines our feature taxonomy: authority markers, third-party validation, and quantified evidence are modelled in detail, while conventional promotional language is expected to contribute little.

Third, **design against position bias and non-independence.** Pfrommer et al. and Chu & Hou make candidate-order randomisation and query-level grouped splitting non-negotiable; without them the study would measure artefacts.

Fourth, **measure what practitioners buy.** Industry convergence on AI Share of Voice means our outputs must be expressed in that currency, extended with the diagnostic attribution metric that no current tool provides.

The contribution to marketing practice is a decision aid, not a dashboard. Given a brand and category, the system estimates visibility, attributes the shortfall between recognition and content, and returns ranked content actions with estimated gains. This converts a currently unmeasured and intuition-driven budget decision — how much to spend on being known versus on being described well — into an evidenced one, in a market where no measurement currently exists.

---

## 6. References

Abolghasemi, A., Verberne, S. & Azzopardi, L. (2024). *Writing Style Matters: An Examination of Bias and Fairness in Information Retrieval Systems.* arXiv:2411.13173.

Adobe (2026). *How to improve brand visibility in AI search engines.* Adobe Business Blog. https://business.adobe.com/blog/improve-brand-visibility-in-ai-search-engines

Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K. & Deshpande, A. (2024). *GEO: Generative Engine Optimization.* Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 50–61.

Bagga, P. S., Wu, Y., Aggarwal, P. & Deshpande, A. (2025). *E-GEO: A Testbed for Generative Engine Optimization in E-Commerce.* arXiv:2511.20867.

Chen, X., Zhang, Z., Zhu, Y., Tao, M. & Wen, J.-R. (2026). *Auditing Preferences for Brands and Cultures in LLMs.* arXiv:2603.18300.

Chu, X. & Hou, Y. (2026). *Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems.* arXiv:2606.17443.

Echterhoff, J., Liu, Y., Alessa, A., McAuley, J. & Chen, Z. (2024). *Cognitive Bias in Decision-Making with LLMs.* Findings of EMNLP 2024, pp. 12594–12613.

Filandrianos, G., Dimitriou, A., Lymperaiou, M., Thomas, K. & Stamou, G. (2025). *Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations.* Proceedings of EMNLP 2025.

Kamruzzaman, M., Nguyen, H. M. & Kim, G. L. (2024). *"Global is good, local is bad?": Understanding Brand Bias in LLMs.* Proceedings of EMNLP 2024, pp. 12704–12721.

Kumar, A. & Lakkaraju, H. (2024). *Manipulating Large Language Models to Increase Product Visibility.* arXiv:2404.07981.

Lichtenberg, J. M., Buchholz, A. & Schwöbel, P. (2024). *Large Language Models as Recommender Systems: A Study of Popularity Bias.* arXiv:2406.01285.

Lin, W., Wang, Y., Bauer, L., He, Y., Seo, M. & Khashabi, D. (2025). *LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses.* Proceedings of CHI 2025.

Nestaas, F., Debenedetti, E. & Tramèr, F. (2025). *Adversarial Search Engine Optimization for Large Language Models.* Proceedings of ICLR 2025.

Pfrommer, S., Bai, Y., Gautam, T. & Sojoudi, S. (2024). *Ranking Manipulation for Conversational Search Engines.* Proceedings of EMNLP 2024, pp. 9520–9534.

Three Rivers AI Nexus (2026). *What Brands Does Your AI Prefer?* Dataset: `3RAIN/brand-bias-evaluations`, HuggingFace, MIT License.
