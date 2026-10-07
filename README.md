# Procedural Naive Bayes sentiment project

Completed scratch multinomial Naive Bayes project with explicit bag-of-words counts, class priors, Laplace likelihoods, prediction, independent library comparison and text inference. The scratch core uses no scikit-learn classifier or vectorizer. It retains the original procedural count → prior → likelihood structure; log scores repair multiplicative underflow and sparse counts bound memory use. See `ATTRIBUTION.md` for source boundaries.

## Run

Use Python 3.12, install `requirements.txt`, then:

```text
python -m pytest -q
python experiment.py train --dataset /path/to/train.csv --output-dir /path/to/local-results
python experiment.py predict --model-dir /path/to/local-results --text "ordinary flight experience"
```

The local dataset can be CSV or XLSX. Default columns are `Tweet` and `Sentiment`; explicitly change them with `--text-column` and `--label-column`. Labels must be negative, neutral or positive. If a workbook has missing text/label rows, `--drop-missing` explicitly removes and records them; otherwise import fails. Optional `--stop-words` reads a local whitespace-separated list. No stop words are removed by default, preserving words such as negation. No tweets, metadata, workbook, fitted vocabulary, model binaries, credentials or media are committed. Keep model output outside the publication tree.

## Fresh experiment

The original PA3 train/test files and stop-word list were not recovered. Fresh verification uses the separately retained airline-sentiment workbook with explicit `text` and `airline_sentiment` columns and `--drop-missing`. This is a replacement-data experiment, not a claimed reproduction of the unavailable PA3 result. The workbook fingerprint and every exclusion count are in `metrics.json`. No raw text or user metadata is included in the metrics.

Cleaning removes usernames, URLs, digits and punctuation and normalizes case; this can lose useful context. Cleaned duplicate texts are deduplicated before splitting, and conflicting label groups/empty cleaned text are excluded with recorded counts. Stratified 60% train / 20% validation / 20% test partitions share no normalized text. The vocabulary is built from training text only, capped at 20,000 frequent words. Validation macro F1 selects smoothing alpha from 0.1, 0.5, 1 and 2; held-out labels never select it.

| Model | Held-out accuracy | Macro F1 |
|---|---:|---:|
| manual_multinomial | 77.07% | 0.686 |
| sklearn_multinomial | 77.07% | 0.686 |
| sklearn_one_vs_rest | 77.21% | 0.683 |

The direct scikit-learn multinomial model independently matches the manual log probabilities. A separate one-versus-rest multinomial comparison satisfies the original project's library section; it has different probability semantics. The saved manual model was reloaded and reproduced all held-out predictions. These are single-split experimental results, not a general-purpose sentiment service, calibrated confidence or validated performance outside this corpus.

Inference reports an `uncalibrated_model_probability`, not certainty. Unknown tokens are ignored; empty/out-of-vocabulary inputs reduce to class priors. Negation, sarcasm and context remain weaknesses of word-count independence. No claim that unseen example phrases are correct is made without labels.

## Notebook and verification

The notebook follows setup → scratch implementation check → experiment → results. Its small mathematical fixture is distinct from measured tweet accuracy; full retraining uses the local dataset command. Outputs are stripped. Tests compare counts/priors/likelihoods/log probabilities with an independent library, check long-document stability, vocabulary leakage, noncontiguous classes, rejected inputs, explicit exclusions, complete training/reload and the prediction command. Supplied prompts are in `ATTRIBUTION.md`; platform grading is not claimed. See `VERIFICATION.json`.
