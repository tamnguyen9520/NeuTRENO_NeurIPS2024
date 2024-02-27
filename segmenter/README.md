This folder contains the source code for the ADE20K Image Segmentation task in [Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals]

It is built on this [repository](https://github.com/rstrudel/segmenter)

## Dataset Setup

Please find the instruction for data download and preprocessing on this [repository](https://github.com/rstrudel/segmenter).

## Scripts
To reproduce the result for segmentation task, it is required to have the deit and neutreno backbone training for the imagenet classification task (please see the DeiT folder).
Run the following script to train the models
  ```angular2html
  bash run.sh
  ```