# End-to-end Bi-grained Contrastive Learning for Multi-face Forgery Detection

This is the official implementation of COMICS: End-to-end Bi-grained Contrastive Learning for Multi-face Forgery Detection


## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).


Then build AdelaiDet with:

```
git clone https://github.com/zhangconghhh/COMICS.git
cd COMICS
python setup.py build develop
```


## Quick Start


To train a model with COMICS, first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
sh train_comics.sh
```
To evaluate the model after training, run:

```
sh eval_comics.sh
```

## Cite the paper
If this work is helpful to you, please cite it as:</p>
```
@ARTICLE{comics_zhanng,
  author={Zhang, Cong and Qi, Honggang and Wang, Shuhui and Li, Yuezun and Lyu, Siwei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={COMICS: End-to-End Bi-Grained Contrastive Learning for Multi-Face Forgery Detection},
  year={2024},
  volume={34},
  number={10},
  pages={10223-10236},
  doi={10.1109/TCSVT.2024.3405563}}
```

## Acknowledgements

The codes are modified from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet). Thanks for their open source.
