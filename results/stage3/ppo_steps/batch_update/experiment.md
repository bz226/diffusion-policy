# DPPO batch-average gradient and the turning of the extra mean

**Status: stopped at G3; downstream analysis unexecuted.** Does averaging DPPO's 20 gradients reduce noise while retaining the epoch-1 mean direction, and how does the reuse-induced extra mean turn? Two group-versus-batch covariance comparisons failed interval agreement: step 19 at 32 updates (ratio 1.118 [1.020, 1.217]) and step 9 at 63 updates (0.962 [0.931, 0.993]). The other 61 covariance checks and all 63 squared-mean checks passed. See the [failure report](REPORT.md).

The batch average is the raw actor-loss gradient G/20, not the optimizer's actual parameter displacement. Epoch 1 is a first-order on-policy reference because its four gradients are evaluated at sequentially moved parameters. The Adam comparison holds the checkpoint preconditioner fixed; actual AdamW changes its moments during the batch.

The approved budget is 30 minutes of local computation, four CPU threads and 8 GiB RAM. No scheduler jobs, GPUs, rollouts or new gradients are allowed. G1–G5 verify inputs, stored sum consistency, group-versus-batch estimators, the reused Gram matrix and preservation of every pre-existing file. A gate failure stops this analysis.

G1, G2 and G4 passed; G5 records preservation of all 2,241 existing files. No figures, frozen-Adam comparisons or Q1–Q6 results were produced. The analysis and plotting code remains unexecuted. Existing scientific outputs and active collections are untouched. Next: await an explicit revised instruction; no automatic continuation or curation is authorized.

[Methods](methods.md) · [Approved scope](provenance/approval.json)
