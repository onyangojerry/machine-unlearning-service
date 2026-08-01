# Selective Retraining

## Hypothesis

Single-record and same-shard deletion requests will require retraining
approximately 20% of the five-shard ensemble. A distributed 1% request
may touch every shard and eliminate most of the computational advantage.

## Integrity requirement

Every unaffected shard must have an identical SHA-256 artifact hash
before and after selective unlearning.

## Measurements

- Affected shard count
- Retraining fraction
- Selective-retraining duration
- Full-retraining duration
- Observed speedup
- Test utility after deletion
- Behavior relative to exact retraining
- Unaffected artifact hash equality