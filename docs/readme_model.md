# Model

## Model decision summary

The production bundle uses a sparse TF-IDF pipeline with multinomial
LogisticRegression. The final classifier is trained on the deduplicated
known-label pool only; the `other` folder is excluded from training and
kept as an out-of-distribution holdout for fallback checks.

After loading and exact-deduplicating the dataset, there are 992 known
documents and 6 `other` holdout documents. The fixed training path uses
a stratified 70/15/15 split: 694 train, 149 validation, and 149 test
documents. Validation is used only for confidence threshold selection;
test is used only for final metrics.

Model selection in `notebooks/02_model_experiments.ipynb` uses 5-fold
stratified cross-validation rather than a single split because the
corpus is small. That gives a more stable model comparison. The final
scripts then use the fixed split described above for reproducibility,
threshold tuning, and held-out reporting.

| Model | CV accuracy | CV macro F1 | Median ms/doc | Batch docs/sec | Size MB |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogisticRegression | 0.9788 | 0.9789 | 0.26 | 6792.06 | 3.28 |
| TF-IDF + MultinomialNB | 0.9778 | 0.9776 | 0.43 | 6721.01 | 5.80 |
| TF-IDF + Calibrated LinearSVC | 0.9819 | 0.9818 | 1.92 | 6187.01 | 13.36 |

The calibrated LinearSVC is marginally ahead on cross-validated macro
F1, but the difference is less than half a point and comes with a larger
artifact and slower single-document inference. MultinomialNB is fast,
but its confidence behavior is too brittle for threshold routing. The
LogisticRegression model is the best balance: simple, fast, compact,
and accurate enough for this case study.

Final held-out test metrics for the trained LogisticRegression bundle:

| Metric | Value |
|---|---:|
| Accuracy | 0.9933 |
| Macro F1 | 0.9933 |
| Weighted F1 | 0.9933 |

The only raw-label test error is a `space` document predicted as
`graphics`; details are in `reports/misclassified_examples.csv` and
`notebooks/03_error_analysis.ipynb`.

## `other` fallback and threshold trade-off

`other` is not a trained class. The model predicts one of the ten known
labels, then `src.predict.predict_text` routes low-confidence documents
to the fallback label. This avoids teaching the classifier that six
miscellaneous files represent the universe of out-of-distribution text.

The final thresholds stored in the bundle are:

| Threshold | Value |
|---|---:|
| `auto_accept` | 0.90 |
| `manual_review` | 0.70 |
| `other` | 0.15 |

The low `other` threshold is deliberate. LogisticRegression is accurate
on this dataset, but many correct predictions have diffuse probability
mass because there are ten classes and short documents. A high fallback
threshold would route too many known-class documents to `other`.

Validation threshold trade-off:

| `other` threshold | Known docs routed to `other` | Known misroute % | Other holdout caught |
|---:|---:|---:|---:|
| 0.10 | 0/149 | 0.00% | 0/6 |
| 0.15 | 3/149 | 2.01% | 6/6 |
| 0.20 | 14/149 | 9.40% | 6/6 |
| 0.35 | 71/149 | 47.65% | 6/6 |
| 0.55 | 129/149 | 86.58% | 6/6 |

The chosen value, 0.15, catches all six `other` examples while keeping
known-class misroutes under 5% on validation. On the held-out test set,
the same threshold routes 2/149 known documents to `other` and catches
6/6 of the OOD holdout.

The **primary, statistically-backed** selection criterion is the
known-label misroute guardrail (≤5% on 149 validation documents). The
6/6 OOD catch is a **secondary smoke check** only: with six files it
cannot bound the false-positive/true-positive trade-off, so it must not
be read as a headline OOD recall result.

### Leave-one-class-out OOD probe

To estimate OOD recall with real statistical power but no new labels,
`evaluate.py` offers an opt-in leave-one-class-out probe (`--loco`, off
by default because it re-trains the model once per class and the core
reports do not need it). Each known class is held out of training in
turn and its documents are treated as pseudo-OOD, while a held-in
validation slice of the remaining classes measures the known-misroute
side. Pooling all ten folds yields a curve over **992** pseudo-OOD
documents (`reports/ood_loco_curve.csv`, regenerated with
`python -m src.evaluate --loco`):

| `other` threshold | Known misroute % | Pseudo-OOD caught % |
|---:|---:|---:|
| 0.10 | 0.00% | 0.0% |
| 0.15 | 0.75% | 27.6% |
| 0.20 | 8.51% | 81.4% |
| 0.25 | 15.90% | 96.1% |
| 0.30 | 29.70% | 99.6% |

This is the honest picture the six-file holdout cannot give: at the
shipped 0.15 the model catches only ~28% of pseudo-OOD documents, not the
100% the smoke check suggests. Catching the bulk of OOD traffic would
require ~0.20, which breaches the 5% misroute guardrail — so 0.15 is the
right *guardrail-constrained* choice, not a high-recall OOD detector. A
production rollout should treat offline OOD recall as a lower bound and
lean on post-hoc monitoring (alerting on the live `fallback_other` and
low-confidence rate) rather than over-trusting any offline estimate.

## Confidence range report

`reports/confidence_ranges.csv` summarizes validation accuracy by top
probability bucket:

| Confidence range | Total | Correct | Accuracy |
|---|---:|---:|---:|
| [0.0, 0.5) | 118 | 114 | 0.9661 |
| [0.5, 0.6) | 18 | 18 | 1.0000 |
| [0.6, 0.7) | 11 | 11 | 1.0000 |
| [0.7, 0.8) | 1 | 1 | 1.0000 |
| [0.8, 0.9) | 1 | 1 | 1.0000 |
| [0.9, 1.0] | 0 | 0 | n/a |

The key point is that low confidence does not mean the model is usually
wrong. Most validation documents land below 0.5 confidence and still
have 96.61% accuracy. The confidence policy should therefore be read as
a conservative routing mechanism, not as a claim that probabilities are
perfectly calibrated.

## Scaling to millions of documents

The trained bundle is 2.07 MB and runs CPU-only sparse linear inference.
Measured on the local evaluation run:

| Benchmark | Value |
|---|---:|
| Median single-document latency | 0.3282 ms |
| p95 single-document latency | 0.5489 ms |
| Batch throughput | 4876.28 docs/sec |
| Batch size used | 149 docs |
| Serialized bundle size | 2.07 MB |

This architecture scales well for high-volume document classification:
the API is stateless, the model is read-only in memory, and replicas can
be added horizontally behind a load balancer. For bulk backfills, a
batch endpoint would improve throughput further by amortizing HTTP and
JSON overhead across many documents.

A transformer or LLM API could improve semantic robustness on harder
OOD cases, but it would be materially more expensive. For this topical
news-style dataset, sparse linear inference is a strong first
production baseline: fast enough for millions of documents on commodity
CPU workers, easy to retrain, easy to inspect, and small enough that
each API worker can load its own copy.
