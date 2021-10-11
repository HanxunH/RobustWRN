# Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks

Code for NeurIPS 2021 Paper ["Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks"](https://arxiv.org/abs/2110.03825) by Hanxun Huang, Yisen Wang, Sarah Monazam Erfani, Quanquan Gu, James Bailey, Xingjun Ma


---
## Robust Configurations for WideResNet (WRN-34-R)
- Model defined in [models/RobustWideResNet.py](models/RobustWideResNet.py)
```python
def RobustWideResNet34(num_classes=10):
    # WRN-34-R configurations
    return RobustWideResNet(
        num_classes=num_classes, channel_configs=[16, 320, 640, 512],
        depth_configs=[5, 5, 5], stride_config=[1, 2, 2], stem_stride=1,
        drop_rate_config=[0.0, 0.0, 0.0], zero_init_residual=False,
        block_types=['basic_block', 'basic_block', 'basic_block'],
        activations=['ReLU', 'ReLU', 'ReLU'], is_imagenet=False,
        use_init=True)
```

---
## Reproduce results from the paper
- Pretrained Weights for WRN-34-R used in Table 2 available on [Google Drive](https://drive.google.com/drive/folders/1MpE1o0C7VjmYPfCIvGdpjMGHGfjs9yuB?usp=sharing)
- All hyperparameters/settings for each model/method used in Table 2 are stored in configs/*.yaml files.

## Evaluations of the robustness of WRN-34-R
##### WRN-34-R trained with TRADES
Replace PGD with other attacks ['CW', 'GAMA', 'AA'].
```console
python main.py --config_path configs/config-WRN-34-R
               --exp_name /path/to/experiments/folders
               --version WRN-34-R-trades
               --load_best_model --attack PGD --data_parallel
```

##### WRN-34-R trained with TRADES and additional 500k data
Replace PGD with other attacks ['CW', 'GAMA', 'AA'].
```console
python main.py --config_path configs/config-WRN-34-R
               --exp_name /path/to/experiments/folders
               --version WRN-34-R-trades-500k
               --load_best_model --attack PGD --data_parallel
```

## Train WRN-34-R with 500k additional data from scratch
```console
python main.py --config_path configs/config-WRN-34-R
               --exp_name /path/to/experiments/folders
               --version WRN-34-R-trades-500k
               --train --data_parallel
```
---
## CIFAR-10 - Linf AutoAttack Leaderboard using additional 500k data
- **Note**: This is not maintained, please find up-to-date leaderboard is available in [RobustBench](https://robustbench.github.io/).

|#    |paper           |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| *available*| WRN-70-16| 91.10| 65.87| 65.88|
|**2**| **Ours**‡ + EMA| *available*| WRN-34-R| 91.23 | 62.54 | 62.54 |
|**3**| **Ours**‡ | *available*| WRN-34-R| 90.56 | 61.56 | 61.56 |
|**4**| [(Wu et al., 2020a)](https://arxiv.org/abs/2010.01279)‡| *available*| WRN-34-15| 87.67| 60.65| 60.65|
|**5**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)‡| *available*| WRN-28-10| 88.25| 60.04| 60.04|
|**6**| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)‡| *available*| WRN-28-10| 89.69| 62.5| 59.53|
|**7**| [(Sehwag et al., 2020)](https://github.com/fra31/auto-attack/issues/7)‡| *available*| WRN-28-10| 88.98| -| 57.14|
|**8**| [(Wang et al., 2020)](https://openreview.net/forum?id=rklOg6EFwS)‡| *available*| WRN-28-10| 87.50| 65.04| 56.29|
---
## Citation
```
@inproceedings{huang2021exploring,
    title={Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks},
    author={Hanxun Huang and Yisen Wang and Sarah Monazam Erfani and Quanquan Gu and James Bailey and Xingjun Ma},
    booktitle={NeurIPS},
    year={2021}
}
```
---
## Part of the code is based on the following repo:
  - MART: https://github.com/YisenWang/MART
  - TREADES: https://github.com/yaodongyu/TRADES
  - RST: https://github.com/yaircarmon/semisup-adv
  - AutoAttack: https://github.com/fra31/auto-attack
  - GAMA: https://github.com/val-iisc/GAMA-GAT
