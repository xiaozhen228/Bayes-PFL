### Bayes-PFL （Accepted by CVPR 2025）
![](figures/framework.png)

**Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection**

Zhen Qu, Xian Tao, Xinyi Gong, ShiChen Qu, Qiyu Chen, Zhengtao Zhang, Xingang Wang, Guiguang Ding

[Paper link](https://arxiv.org/pdf/2503.10080)

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Environments](#environments)
* [📊 Data Preparation](#data-preparation)
* [🚀 Run Experiments](#run-experiments)
* [🔗 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)
* [📜 License](#license)

## Introduction
**This repository contains source code for VCP-CLIP implemented with PyTorch.** 


Recently, vision-language models (e.g. CLIP) have demonstrated remarkable performance in zero-shot anomaly detection (ZSAD). By leveraging auxiliary data during training, these models can directly perform cross-category anomaly detection on target datasets, such as detecting defects on industrial product surfaces or identifying tumors in organ tissues. Existing approaches typically construct text prompts through either manual design or the optimization of learnable prompt vectors. However, these methods face several challenges: 1) handcrafted prompts require extensive expert knowledge and trial-and-error; 2) single-form learnable prompts struggle to capture complex anomaly semantics; and 3) an unconstrained prompt space limits generalization to unseen categories. To address these issues, we propose Bayesian Prompt Flow Learning (Bayes-PFL), which models the prompt space as a learnable probability distribution from a Bayesian perspective. Specifically, a prompt flow module is designed to learn both imagespecific and image-agnostic distributions, which are jointly utilized to regularize the text prompt space and improve the model’s generalization on unseen categories. These learned distributions are then sampled to generate diverse text prompts, effectively covering the prompt space. Additionally, a residual cross-model attention (RCA) module is introduced to better align dynamic text embeddings with fine-grained image features. Extensive experiments on 15 industrial and medical datasets demonstrate our method’s superior performance.


## Environments
Create a new conda environment and install required packages.
```
conda create -n Bayes_PFL python=3.9
conda activate VCP_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Experiments are conducted on a NVIDIA RTX 3090.**


## Data Preparation
 
#### MVTec-AD and VisA 

> **1、Download and prepare the original [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) datasets to any desired path. The original dataset format is as follows:**

```
path1
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png
```

```
path2
├── visa
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
    ├── split_csv
        ├── 1cls.csv
        ├── 1cls.xlsx
```

> **2、Standardize the MVTec-AD and VisA datasets to the same format and generate the corresponding .json files.**

- run **./dataset/make_dataset_new.py** to generate standardized datasets **./dataset/mvisa/data/visa** and **./dataset/mvisa/data/mvtec**
- run **./dataset/make_meta.py** to generate **./dataset/mvisa/data/meta_visa.json** and **./dataset/mvisa/data/meta_mvtec.json** (This step can be skipped since we have already generated them.)

The format of the standardized datasets is as follows:

```
./datasets/mvisa/data
├── visa
    ├── candle
        ├── train
            ├── good
                ├── visa_0000_000502.bmp
        ├── test
            ├── good
                ├── visa_0011_000934.bmp
            ├── anomaly
                ├── visa_000_001000.bmp
        ├── ground_truth
            ├── anomaly1
                ├── visa_000_001000.png
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── mvtec_000000.bmp
        ├── test
            ├── good
                ├── mvtec_good_000272.bmp
            ├── anomaly
                ├── mvtec_broken_large_000209.bmp
        ├── ground_truth
            ├── anomaly
                ├── mvtec_broken_large_000209.png

├── meta_mvtec.json
├── meta_visa.json
```

## Run Experiments


## Citation
Please cite the following paper if the code help your project:

```bibtex
@article{qu2025bayesian,
  title={Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection},
  author={Qu, Zhen and Tao, Xian and Gong, Xinyi and Qu, Shichen and Chen, Qiyu and Zhang, Zhengtao and Wang, Xingang and Ding, Guiguang},
  journal={arXiv preprint arXiv:2503.10080},
  year={2025}
```

## Acknowledgements


## License
The code and dataset in this repository are licensed under the [MIT license](https://mit-license.org/).