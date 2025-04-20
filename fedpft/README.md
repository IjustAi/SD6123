---
title: Parametric Feature Transfer, One-shot Federated Learning with Foundation Models
url: https://arxiv.org/abs/2402.01862
dataset: fed-isic2019 
---

## Result

Run FedPFT on fed-isic2019 dataset for a Mac with an Apple M-series chip.
```bash

python -m fedpft.main dataset=isic device=mps num_clients=4 num_rounds=1 batch_size=32 num_gpus=0

```
Compare with FedAvg

![](_static/fed_comparison.png)
![](_static/Training_log_pft.png)
![](_static/Training_log_avg.png)
