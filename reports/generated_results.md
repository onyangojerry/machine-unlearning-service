# Generated experimental results

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
| Observed speedup | 4.114894× |
| Test prediction disagreement | 0.000000 |
| Test mean probability gap | 0.000000 |
| Unaffected hashes unchanged | Yes |

## Membership-inference evaluation

| Measurement | Original | Exact reference |
|---|---:|---:|
| Attack ROC-AUC | 0.470382 | 0.470428 |
| Forget-set predicted-member rate | 0.997442 | 0.997442 |
