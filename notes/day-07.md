# Day 7 — Release Audit

## Evidence inventory

| Claim | Artifact | Status |
|---|---|---|
| Baseline utility measured | artifacts/baseline_metrics.json | PASS |
| Forget set is deterministic | artifacts/forget_manifest.json | PASS |
| Exact retain-only model exists | artifacts/reference_model.joblib | PASS |
| Exact record accounting holds | artifacts/exact_unlearning_results.json | PASS |
| Sharded ensemble contains five models | artifacts/shards/ | PASS |
| Selective result matches reference | artifacts/selective_single/selective_results.json | PASS |
| Unaffected hashes are unchanged | artifacts/selective_single/selective_results.json | PASS |
| Retraining speedup is measured | artifacts/selective_single/selective_results.json | PASS |
| Membership attack quality is reported | artifacts/privacy_results.json | PASS |
| CI runs unit tests | .github/workflows/ci.yml | PASS |
| Full workflow is reproducible | scripts/run_pipeline.py | PASS |

## Claim-language review

Use:
- exact retraining reference
- SISA-style sharded unlearning
- empirical membership-inference signal
- observed runtime speedup
- byte-identical unaffected artifacts

Avoid:
- certified deletion
- complete privacy
- full SISA
- guaranteed speedup
- proof that all information was removed

## Week 1 retrospective

### What worked

- Stable record identifiers made deletion requests auditable.
- Exact retraining supplied a trustworthy counterfactual baseline.
- Architecture-matched comparison isolated selective-retraining correctness.
- Artifact hashes demonstrated that unaffected shard files were preserved.
- Central configuration and artifact contracts reduced silent experiment drift.

### What was harder than expected

- Different deletion workloads could not be combined into one privacy claim.
- Runtime results varied more than predictive metrics.
- A uniformly distributed deletion set could touch every shard and remove the
  expected computational advantage.
- One deleted record was insufficient for statistical privacy conclusions.

### What remains incomplete

- Slice checkpointing
- Stronger membership-inference attacks
- Larger and repeated deletion workloads
- Parallel shard retraining
- Formal deletion certification
- Evaluation on nonlinear or deep models

### Main lesson

Machine unlearning is not demonstrated by low accuracy on deleted records.
The central engineering requirement is that the resulting system behave like
an appropriate never-trained reference while producing an auditable deletion
trail.