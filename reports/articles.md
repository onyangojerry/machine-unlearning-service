# Can a Model Efficiently Forget a User?

## Experimental setup

We use version 2 of the Adult Income dataset and a deterministic
80/20 stratified split. Every record receives a stable identifier
before splitting. The original model combines median numerical
imputation, feature scaling, categorical imputation, one-hot encoding
and logistic regression in one serialized pipeline.

A seeded uniform sampler selects 1% of the training records as the
forget set. All later unlearning methods receive this exact manifest.

## Baseline results

| Measurement | Result |
|---|---:|
| Test accuracy | 0.8523902139420616 |
| Test F1 | 0.6563393708293613 |
| Test ROC-AUC | 0.9042302959684185 |
| Test log loss | 0.32108308976974026 |
| Training time | 0.3593632999691181 seconds |
| Training records | 39073 |
| Forget records | 391 |

## Initial interpretation

The original model establishes predictive utility before deletion.
These results do not demonstrate forgetting. Subsequent experiments
will compare retrained and selectively unlearned models against this
baseline.