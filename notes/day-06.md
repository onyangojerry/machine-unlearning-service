# Day 6 — Reproducibility and CI

## Reproducibility contract

Every experiment must identify its dataset version, configuration, random
seed, input artifacts and generated outputs. An experiment is reproducible
only when another environment can reconstruct its non-runtime results.

## Test layers

Unit tests use local deterministic fixtures and run on every pull request.

Integration tests may download the Adult dataset, train models or inspect
generated artifacts. They run separately so an external service outage does
not obscure a code regression.

## CI hypothesis

A clean environment should be able to install the package and run all
network-independent tests without setting PYTHONPATH manually.

# questions to consider;

Why is a random seed insufficient for full reproducibility?
Why should an unknown configuration key cause a failure?
Why should ordinary PR tests avoid external dataset downloads?
Which results should remain deterministic, and which may vary?
Why must artifact dependencies be executed in order?

baseline
  ├── original model
  ├── baseline metrics
  └── forget manifest
        ↓
exact reference
        ↓
sharded ensemble
        ↓
selective plan
        ↓
selective unlearning
        ↓
privacy evaluation