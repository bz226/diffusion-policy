# Seed-0 interim comparison

Evaluation is from iteration 70 (63 prior updates); training return and J_disc are from iteration 69 (62 prior updates). Every value comes from that exact iteration. Historical DPPO and without_PPO did not log J_disc.

| Method | Eval, itr 70 | Train return, itr 69 | J_disc, itr 69 |
| --- | --- | --- | --- |
| DPPO | 4549.2 | 4261.8 | not logged |
| without_PPO | 4547.6 | 4215.7 | not logged |
| without_PPO, lr 1e-3 | 3359.1 | 3256.6 | not logged |
| DPPO, 1 epoch | 4522.5 | 4257.3 | 1496.2 |
| DPPO, 1 epoch, lr 5e-4 | 4419.3 | 4174.3 | 1476.2 |
| DPPO, 2 epochs | 4640.4 | 4424.3 | 1537.2 |
| DPPO, 2 epochs, lr 2.5e-4 | 4674.8 | 4425 | 1539.6 |

Against DPPO at the same iterations: DPPO, 1 epoch: eval gap -26.631, train gap -4.4915; DPPO, 1 epoch, lr 5e-4: eval gap -129.82, train gap -87.478; DPPO, 2 epochs: eval gap 91.269, train gap 162.53; DPPO, 2 epochs, lr 2.5e-4: eval gap 125.58, train gap 163.22.

Against without_PPO at the same iterations: DPPO, 1 epoch: eval gap -25.058, train gap 41.594; DPPO, 1 epoch, lr 5e-4: eval gap -128.25, train gap -41.392; DPPO, 2 epochs: eval gap 92.842, train gap 208.62; DPPO, 2 epochs, lr 2.5e-4: eval gap 127.15, train gap 209.3.

These are interim observations from one seed; they do not determine the final comparison.
