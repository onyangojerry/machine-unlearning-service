# Auditable Machine-Unlearning Service

A reproducible tabular machine-unlearning experiment comparing exact
retraining with SISA-style selective shard retraining on the Adult Income
dataset.

## Why this project exists

Deleting a row from a database does not remove its influence from an
already-trained model. This project evaluates whether retraining can remove
that record's training participation while preserving utility and reducing
deletion cost.

## System architecture

Dataset and stable record IDs
        |
        v
Deterministic train/test split
        |
        +--> Original single model
        |
        +--> Exact retain-only reference
        |
        +--> Five isolated shard models
                    |
Deletion manifest -> affected-shard routing
                    |
                    v
             Selective retraining
                    |
                    v
       Utility, behavior, cost and privacy evaluation

## Methods implemented

- Exact retain-only retraining
- Deterministic SHA-256 shard assignment
- Independent preprocessing and classification per shard
- Mean-probability ensemble aggregation
- Selective retraining of deletion-affected shards
- Architecture-matched full-retraining reference
- Confidence-based membership-inference evaluation
- SHA-256 integrity verification for unaffected artifacts

This is a SISA-style implementation. Slice checkpointing is not yet
implemented.

## Key results

## Utility

| Metric | Original | Exact reference | Sharded ensemble |
|---|---:|---:|---:|
| accuracy | 0.852390 | 0.852288 | 0.853004 |
| f1 | 0.656339 | 0.656183 | 0.658258 |
| roc_auc | 0.904230 | 0.904225 | 0.904652 |
| log_loss | 0.321083 | 0.321123 | 0.320237 |

## Selective unlearning

| Measurement | Result |
|---|---:|
| Affected shards | 1 |
| Selective retraining seconds | 0.077546 |
| Full sharded retraining seconds | 0.319094 |
| Observed speedup | 4.114894Ã— |
| Test prediction disagreement | 0.000000 |
| Test mean probability gap | 0.000000 |
| Unaffected hashes unchanged | Yes |

## Membership-inference evaluation

| Measurement | Original | Exact reference |
|---|---:|---:|
| Attack ROC-AUC | 0.470382 | 0.470428 |
| Forget-set predicted-member rate | 0.997442 | 0.997442 |



# Quick Start

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt

## Run the complete experiment

python scripts\run_pipeline.py
python scripts\audit_project.py

## Run Tests

pytest -m "not integration" -v
pytest -v
ruff check src tests scripts

## Inspect MLFlow

mlflow ui --backend-store-uri ./mlruns



Open http://127.0.0.1:5000.

## Reproducibility

The experiment uses a versioned configuration, stable record identifiers,
seeded splits, deterministic shard assignment and seeded bootstrap
resampling. Runtime fields are excluded from equality comparisons because
they depend on hardware and system load.

## Limitations
Slice checkpoints are not implemented.
The classifier is a linear tabular baseline.
The privacy evaluation covers one black-box confidence attack.
One-record privacy results are descriptive.
Behavioral similarity and attack resistance are empirical evidence, not
a formal deletion certificate.
## References
Bourtoule et al., Machine Unlearning
Shokri et al., Membership Inference Attacks Against Machine Learning Models
Yeom et al., Privacy Risk in Machine Learning

Add both figures below the results section:

```markdown
![Probability shift on forgotten records](reports/figures/forget_probability_shift.png)

![Membership-inference comparison](reports/figures/membership_inference_rates.png)
6. Complete the technical article — 20 minutes

In reports/article.md:

Replace every TODO using JSON artifacts or generated_results.md.
Retain enough decimal precision to reproduce differences.
Explicitly distinguish the single-model and sharded architectures.
Report runtime as machine-dependent.
Interpret privacy only after reporting attack ROC-AUC.
Keep single-record privacy evidence descriptive.
State that exact retraining defines the counterfactual reference.
Use “SISA-style” consistently.

## Search for incomplete language:

rg -n "TODO|TBD|FIXME|full SISA|proved privacy|certified deletion" `
  README.md `
  reports `
  notes

Expected output: no unresolved placeholders and no unsupported claims.

Recommended article conclusion:



## Conclusion

Exact retraining established the behavior of a model that never trained on
the deletion records. The SISA-style system reproduced its
architecture-matched reference by retraining only affected shards, while
unaffected model artifacts remained byte-identical.

The measured computational benefit depended on how many shards the request
touched. This confirms that deletion locality, rather than deletion count
alone, determines the benefit of sharded unlearning.

The membership-inference experiment supplied an additional privacy signal,
but its conclusions are bounded by the attack's discrimination and forget-set
size. The experiment therefore supports an auditable engineering claim, not
a formal guarantee of information removal.