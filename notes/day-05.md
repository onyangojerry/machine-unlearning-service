# Day 5 — Membership-Inference Evaluation

## Threat model

The attacker has black-box access to model probabilities and knows the
correct label for a candidate record. The attacker does not have model
parameters or training logs.

## Attack

The attack uses true-label log-confidence. A threshold is calibrated
using known members and known nonmembers.

## Hypothesis

Forgotten records will have a lower predicted-member rate under the
exact-reference and selectively unlearned models than under the
original model.

## Limitations

- Confidence attacks are not optimal attacks.
- A low attack rate does not prove complete information removal.
- Model calibration affects confidence scores.
- Small forget sets produce uncertain estimates.
- Results may not generalize to stronger white-box attacks.