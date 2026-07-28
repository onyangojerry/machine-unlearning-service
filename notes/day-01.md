# Machine-Unlearning

## Research question

Can selective retraining remove specified training records while
preserving model utility and reducing computational cost relative to
full retraining?

## Hypothesis
SISA-style selective retraining will approach the predictive behavior
of full retraining while requiring less retraining time.

## Forget set

A deterministic collection of training record IDs submitted through a
versioned deletion request.

## Retain set

All original training records except those included in the forget set.

## Reference model

A new model trained from scratch using only the retain set.

## Primary measurements

- Test accuracy and F1
- Prediction agreement with the reference model
- Forget-set loss
- Retraining duration
- Membership-inference attack performance