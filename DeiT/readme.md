# Image Classification - ImageNet

This folder contains the source code for Imagenet classification in [Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals]

It is built on this [repository](https://github.com/facebookresearch/deit/blob/main/README_deit.md)

In this task, the model learns to predict the class of an image, out of 1000 classes.

## Requirements

- python >= 3.7
- python libraries:
```bash
pip install -r requirements.txt
```

## Data preparation

We use the standard ImageNet dataset, you can download it from http://image-net.org/. Validation images are put in labeled sub-folders. The file structure should look like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Scripts
Run the following script to train the models
  ```angular2html
  bash run.sh
  ```


## Reference repositories
- [Transformer-ls](https://github.com/facebookresearch/deit/)
