# SSLMVC: Self-supervised label-driven multi-view collaborative clustering.


Our work has been accepted by Neurocomputing. https://www.sciencedirect.com/science/article/pii/S0925231226018904

## Requirements

- Python 3.12
- PyTorch==2.7.0
- NumPy==2.3.3
- scikit-learn==1.7.2
- SciPy==1.16.2


## Dataset

Please place the dataset in the `data/` directory. 

## Training

To train the model with default parameters:

```bash
python train.py 
```
After running, you will obtain running results similar to the ones shown below：

```
==================================================
1 runs complete! Best results:
ACC: 0.8943
NMI: 0.8142
ARI: 0.7870
PUR: 0.8943
==================================================
```

## Citation

If you use this code or find our work helpful, please cite our paper:

```
@article{ZHAO2026134492,
title = {SSLMVC: Self-supervised label-driven multi-view collaborative clustering},
journal = {Neurocomputing},
volume = {700},
pages = {134492},
year = {2026},
issn = {0925-2312},
author = {Xiaotong Zhao and Limin Chen and Yujie Tian and Hui Wang},
}
```

## License

This code is released for academic research purposes only.


You can contact zhaoxt99@foxmail.com if you have any questions.
