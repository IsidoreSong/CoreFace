## This is the official repository of the paper:

#### ContraFace: Sample-Guided Contrastive Learning for Deep Face Recognition

![image](https://raw.githubusercontent.com/IsidoreSong/ContraFce/master/images/framework.png)

## Evaluation Results

| Methods(%)     | Venue    | LFW           | AgeDB         | CFP-FP        | CALFW         | CPLFW         | Average        |
| -------------- | -------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------- |
| CosFace        | CVPR2018 | 99.81         | 98.11         | 98.12         | 95.76         | 92.28         | 96.816         |
| ArcFace        | CVPR2019 | **99.83**     | 98.28         | 98.27         | 95.45         | 92.08         | 96.782         |
| AFRN           | ICCV2019 | ***`99.85`*** | 95.35         | 95.56         | ***`96.30`*** | **93.48**     | 96.108         |
| MV-Softmax     | AAAI2020 | 99.80         | 97.95         | 98.28         | 96.10         | 92.83         | 96.992         |
| CurricularFace | CVPR2020 | 99.80         | 98.32         | 98.37         | **96.20**     | 93.13         | 97.164         |
| BroadFace      | ECCV2020 | ***`99.85`*** | ***`98.38`*** | ***`98.63`*** | **96.20**     | 93.17         | **97.246**     |
| SCF-ArcFace    | CVPR2021 | 99.82         | 98.30         | 98.40         | 96.12         | 93.16         | 97.160         |
| MagFace        | CVPR2021 | **99.83**     | 98.17         | 98.46         | 96.15         | 92.87         | 97.096         |
| AdaFace        | CVPR2022 | 99.82         | 98.05         | 98.49         | 96.08         | ***`93.53`*** | 97.194         |
| ContraFace     | AAAI2023 | **99.83**     | **98.37**     | **98.60**     | **96.20**     | 93.27         | ***`97.254`*** |

### Model Training

We take MS1MV2 as the training dataset which can be downloaded from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) (MS1M-ArcFace in DataZoo).
Set the `config.output` and `config.rec` in the config/config.py

Our model is trained on 4 NVIDIA A100 GPUs for about 39 hours.

An economic alternative is trained on CASIA-WebFace with ResNet50 as the backbone.

The `config.eval_step` is set to the number of steps in an epoch so the model would be evaluated at the end of every epoch.

The log files could be find in the project and the model file can be downloaded [here](https://pan.baidu.com/s/1flXRLYRvL15HGVyJAKjygQ?pwd=g5by) (with extraction code *g5by*).

You may train a new model with the command below.

```shell
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1235 train.py
```

## Face recognition evaluation

#### Evaluation on LFW, AgeDb-30, CPLFW, CALFW and CFP-FP:

1. download the data from their offical webpages or the bin files from insightface datasets
2. set the config.rec to dataset folder e.g. data/faces_emore
3. set the config.val_targets for list of the evaluation dataset
4. just use the `CallBackVerification` in utils/utils_callbacks.py to evaluate a model
5. the evaluation of all of the five models is less than 100 seconds as we make it parrallel

#### Evaluation on IJB and megaface

You may find the evaluation scripts at [insightface/recognition/\_evaluation\_](https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_).

As the original IJB scripts don't support parameter-only saved pytorch models, we change the code and put them with megaface scripts in eval folder. Examples are located in two sub-folders separately.

The evaluation of IJB-B and IJB-C cost about 15 mins and 33 mins on one 2080-Ti GPU.

The evaluation of megaface needs their linux-only develop suit. It's really time-consuming (2+3 hours on one 2080-Ti) sincethe operation on 1M images.

## Acknowledgement

[GitHub - deepinsight/insightface: State-of-the-art 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface) You may also find more useful information here, especially in their issue.

[GitHub - fdbtrs/ElasticFace: Official repository for ElasticFace: Elastic Margin Loss for Deep Face Recognition](https://github.com/fdbtrs/ElasticFace)

[GitHub - MLNLP-World/SimBiber: A tool for simplifying bibtex with official info](https://github.com/MLNLP-World/Simbiber)

[GitHub - yuchenlin/rebiber: A simple tool to update bib entries with their official information (e.g., DBLP or the ACL anthology).](https://github.com/yuchenlin/rebiber)

[GitHub - MLNLP-World/Paper-Writing-Tips: Paper Writing Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)

## License

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license.
```
