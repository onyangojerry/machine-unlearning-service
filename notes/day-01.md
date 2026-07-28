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



## hypothesis;

A deterministic preprocessing and training pipeline should produce the
same predictions and metrics when trained repeatedly with an identical
dataset, configuration and random seed.

The original model is not an unlearned model. It establishes the
utility, runtime and privacy measurements against which subsequent
models will be evaluated.

## Above Metrics

"accuracy": 0.8523902139420616,
"precision": 0.7411194833153929,
"recall": 0.5889649272882805,
"f1": 0.6563393708293613,
"roc_auc": 0.9042302959684185,
"log_loss": 0.32108308976974026,
"training_seconds": 0.3593632999691181,
"training_records": 39073,
"test_records": 9769

  ## similarly, reproducibility;

 accuracy 0.8523902139420616 0.8523902139420616 True
 f1 0.6563393708293613 0.6563393708293613 True
 log_loss 0.32108308976974026 0.32108308976974026 True
 precision 0.7411194833153929 0.7411194833153929 True
 recall 0.5889649272882805 0.5889649272882805 True
 roc_auc 0.9042302959684185 0.9042302959684185 True
 test_records 9769 9769 True
 training_records 39073 39073 True