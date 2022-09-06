# Distributional Correlation-aware KD

Code and data for Distributional Correlation–Aware Knowledge Distillation for Stock Trading Volume Prediction (ECML-PKDD 22)

The key idea is the turn the regression KD problem into a distributional matching problem:

![Idea Illustration]()

## Setup

We recommand to setup the running enviroment via conda:

```bash
conda create -n dckd python=3.7
conda activate dckd 
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt 
```

## Dataset

We collect the trading data of TPX500, can the original data of all tickets can be downloaded from [Google Drive](https://drive.google.com/file/d/14ekzilkEUXK8MoNtuuf40qKR0OO45A5d/view?usp=sharing)

Unzip the `tpx500.zip` under the project root dir and you can check the `topix500` directory to see the raw data.

## Training Teacher Model 

For distillation, we first train a large teacher model with DeepAR on the whole dataset.

The training can be started by executing:

```bash
sh train_teacher.sh
```

Check the `ar_kd_teacher.py` for corresponding setting parameters like number of model layers.

After training, the best teacher model will be saved at `teacher_ckpt` and we can use it to train the student later.

## Distillation 

Specify the teacher path in the `run_kd.sh` and execute the script for training the student model:

```bash
sh run_kd.sh
```

## Acknowledgement

We thank Zhiyuan Zhang for providing the code base.

If you find this repo and the data helpful, please kindly cite our paper:

```html
@article{Li2022DistributionalCK,
title={Distributional Correlation-Aware Knowledge Distillation for Stock Trading Volume Prediction},
  author={Lei Li and Zhiyuan Zhang and Ruihan Bao and Keiko Harimoto and Xu Sun},
  journal={ArXiv},
  year={2022},
  volume={abs/2208.07232}
}
```
