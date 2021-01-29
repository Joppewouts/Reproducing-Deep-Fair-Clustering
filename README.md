# [Reproducibility study] Deep Fair Clustering for Visual Learning

This repository is the official implementation of [[Reproducibility study] Deep Fair Clustering for Visual Learning](https://openreview.net/forum?id=MhMYW2PqGSH&noteId=V4snwvjEkeh). 
Our work reproduces the results of ["Peizhao Li, Han Zhao, and Hongfu Liu. "Deep Fair Clustering for Visual Learning", CVPR 2020."](https://github.com/brandeis-machine-learning/DeepFairClustering)


## Requirements

To install requirements run:

```setup
pip install -r requirements.txt
```
If `pipenv` is installed it is recommended to run 
```setup
pipenv install -r requirements.txt
```
### Datasets
Any datasets that need to be downloaded should be downloaded automatically, it is also possible to run one of the download scripts listed in `./data`

### Model tracking and hyperparameter tuning

We used [Weights & Biases](https://wandb.ai/) to track and hyperparameter tune our models

## Training

To train the model(s) in the paper, run this command and select the options which are relevant for your use case:

```train
python main.py [-h] [--encoder_path ENCODER_PATH] [--encoder_legacy_path ENCODER_LEGACY_PATH]
               [--encoder_0_legacy_path ENCODER_0_LEGACY_PATH] [--encoder_0_path ENCODER_0_PATH]
               [--encoder_1_legacy_path ENCODER_1_LEGACY_PATH] [--encoder_1_path ENCODER_1_PATH] [--encoder_lr ENCODER_LR]
               [--encoder_bs ENCODER_BS] [--encoder_max_epochs ENCODER_MAX_EPOCHS] [--encoder_type ENCODER_TYPE]
               [--cluster_0_path CLUSTER_0_PATH] [--cluster_1_path CLUSTER_1_PATH] [--cluster_number CLUSTER_NUMBER]
               [--cluster_path CLUSTER_PATH] [--cluster_n_init CLUSTER_N_INIT] [--cluster_max_step CLUSTER_MAX_STEP]
               [--dfc_0_path DFC_0_PATH] [--dfc_1_path DFC_1_PATH] [--dfc_path DFC_PATH] [--dfc_hidden_dim DFC_HIDDEN_DIM]
               [--adv_multiplier ADV_MULTIPLIER] [--dfc_tradeoff DFC_TRADEOFF] [--dec_lr DEC_LR] [--dec_batch_size DEC_BATCH_SIZE]
               [--dec_iters DEC_ITERS] [--dataset DATASET] [--input_height INPUT_HEIGHT] [--digital_dataset DIGITAL_DATASET]
               [--method METHOD] [--iters ITERS] [--lr LR] [--test_interval TEST_INTERVAL] [--bs BS] [--log_dir LOG_DIR] [--gpu GPU]
               [--seed SEED] [--half_tensor]

optional arguments:
  -h, --help            show this help message and exit
  --encoder_path ENCODER_PATH
  --encoder_legacy_path ENCODER_LEGACY_PATH
  --encoder_0_legacy_path ENCODER_0_LEGACY_PATH
  --encoder_0_path ENCODER_0_PATH
  --encoder_1_legacy_path ENCODER_1_LEGACY_PATH
  --encoder_1_path ENCODER_1_PATH
  --encoder_lr ENCODER_LR
  --encoder_bs ENCODER_BS
  --encoder_max_epochs ENCODER_MAX_EPOCHS
  --encoder_type ENCODER_TYPE
  --cluster_0_path CLUSTER_0_PATH
  --cluster_1_path CLUSTER_1_PATH
  --cluster_number CLUSTER_NUMBER
  --cluster_path CLUSTER_PATH
  --cluster_n_init CLUSTER_N_INIT
  --cluster_max_step CLUSTER_MAX_STEP
  --dfc_0_path DFC_0_PATH
  --dfc_1_path DFC_1_PATH
  --dfc_path DFC_PATH
  --dfc_hidden_dim DFC_HIDDEN_DIM
  --adv_multiplier ADV_MULTIPLIER
  --dfc_tradeoff DFC_TRADEOFF
  --dec_lr DEC_LR
  --dec_batch_size DEC_BATCH_SIZE
  --dec_iters DEC_ITERS
  --dataset DATASET
  --input_height INPUT_HEIGHT
  --digital_dataset DIGITAL_DATASET
  --method METHOD
  --iters ITERS
  --lr LR
  --test_interval TEST_INTERVAL
  --bs BS
  --log_dir LOG_DIR
  --gpu GPU
  --seed SEED
  --half_tensor
```

## Evaluation

To evaluate the results we provide in our paper, launch the provided notebook:

## Pre-trained Models

You can download pretrained models/encoders here:

- [Link to pretrained model/encoder files](https://amsuni-my.sharepoint.com/:f:/g/personal/rodrigo_alejandro_chavez_mulsa_student_uva_nl/ErT0dJvR5whBpwrHFsVmWRsBMFWct2DZst8oN3OL1lp__A)

## Contributing

```text
MIT License

Copyright (c) 2020 Wouts, Sevenster, van de Kar, Alejandro Chávez Mulsa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgements

The DFC methods that we provide have been proposed by 

> "Deep Fair Clustering for Visual Learning" \
> Peizhao Li, Han Zhao, and Hongfu Liu.  
> CVPR 2020 (https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Deep_Fair_Clustering_for_Visual_Learning_CVPR_2020_paper.html)

and full credit for these methods goes to them.

The code that we provide is largely based on their code, which is available in their Github repository: \
https://github.com/brandeis-machine-learning/DeepFairClustering

Tests are for the biggest part based on: \
https://github.com/vlukiyanov/pt-dec