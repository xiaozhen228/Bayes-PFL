### Bayes-PFL
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
**This repository contains source code for Bayes-PFL implemented with PyTorch （Accepted by CVPR 2025）.** 


Recently, vision-language models (e.g. CLIP) have demonstrated remarkable performance in zero-shot anomaly detection (ZSAD). By leveraging auxiliary data during training, these models can directly perform cross-category anomaly detection on target datasets, such as detecting defects on industrial product surfaces or identifying tumors in organ tissues. Existing approaches typically construct text prompts through either manual design or the optimization of learnable prompt vectors. However, these methods face several challenges: 1) handcrafted prompts require extensive expert knowledge and trial-and-error; 2) single-form learnable prompts struggle to capture complex anomaly semantics; and 3) an unconstrained prompt space limits generalization to unseen categories. To address these issues, we propose Bayesian Prompt Flow Learning (Bayes-PFL), which models the prompt space as a learnable probability distribution from a Bayesian perspective. Specifically, a prompt flow module is designed to learn both imagespecific and image-agnostic distributions, which are jointly utilized to regularize the text prompt space and improve the model’s generalization on unseen categories. These learned distributions are then sampled to generate diverse text prompts, effectively covering the prompt space. Additionally, a residual cross-model attention (RCA) module is introduced to better align dynamic text embeddings with fine-grained image features. Extensive experiments on 15 industrial and medical datasets demonstrate our method’s superior performance.


## Environments
Create a new conda environment and install required packages.
```
conda create -n Bayes_PFL python=3.9
conda activate Bayes_PFL
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
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

- run **./dataset/make_dataset.py** to generate standardized datasets **./dataset/mvisa/data/visa** and **./dataset/mvisa/data/mvtec**
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
#### Other Datasets
Updating...


## Run Experiments
#### Prepare the pre-trained weights
> 1、 Download the CLIP weights pretrained by OpenAI [[ViT-L-14-336](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)(default),  [ViT-B-16-224](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), [ViT-L-14-224](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) to **./pretrained_weight/**

> 2、If you are interested, please download one of the pre-trained weights of our Bayes-PFL to **./bayes_weight/**. "train_visa.pth" indicates that the auxiliary training dataset is VisA, which you can utilize to test any products outside of the VisA dataset [[train_visa.pth]](https://drive.google.com/file/d/1rNs_rdTmrg4JshmKHotq6AN1gqTpPjvm/view?usp=drive_link),  and vice versa [[train_mvtec.pth]](https://drive.google.com/file/d/1EHa4jPi7r8jmRVURoZ4yH2-jqDt6K1Ni/view?usp=drive_link). Note that if you use our pre-trained weights, you must use [[ViT-L-14-336](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)] as a default backbone.


#### Training on the seen products of auxiliary datasets

> bash train.sh

#### Testing and visualizing on the unseen products

> bash test.sh

Note that we perform auxiliary training on one industrial dataset and directly infer on other industrial and medical datasets. Since the categories in VisA do not overlap with those in the other datasets, we use VisA as the auxiliary training set. To assess VisA itself, we fine-tune our model on the MVTec-AD dataset.
## Citation
Please cite the following paper if the code help your project:

```bibtex
@article{qu2025bayesian,
  title={Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection},
  author={Qu, Zhen and Tao, Xian and Gong, Xinyi and Qu, Shichen and Chen, Qiyu and Zhang, Zhengtao and Wang, Xingang and Ding, Guiguang},
  journal={arXiv preprint arXiv:2503.10080},
  year={2025}
}
```

## Acknowledgements
We thank the great works [WinCLIP(zqhang)](https://github.com/zqhang/Accurate-WinCLIP-pytorch), [WinCLIP(caoyunkang)](https://github.com/caoyunkang/WinClip), [CLIP-AD](https://github.com/ByChelsea/CLIP-AD), [VCP-CLIP](https://github.com/xiaozhen228/VCP-CLIP), [APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN), [AdaCLIP](https://github.com/caoyunkang/AdaCLIP) and [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) for assisting with our work.

## License
The code and dataset in this repository are licensed under the [MIT license](https://mit-license.org/).