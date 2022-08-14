#!/usr/bin/env bash

#python -u ijb_11.py --model-prefix ./pretrained_models/r100-arcface/model --model-epoch 1 --gpu 0 --target IJBC --job arcface > ijbc_11.log 2>&1 &

#python -u ijb_1n.py --model-prefix ./pretrained_models/r100-arcface/model --model-epoch 1 --gpu 0 --target IJBB --job arcface > ijbb_1n.log 2>&1 &

#python ijb_evals.py -A


export CUDA_VISIBLE_DEVICES=1

python /workspace/_evaluation_/ijb/ijb_evals.py -s IJBC -m /dataset/model/AdaFace_Emore_d46_5505-W/272928model.pth -d /dataset/IJB/IJB_release

#python /workspace/_evaluation_/ijb/ijb_evals.py -s IJBC -m /dataset/model/AdaFace_Emore_d46_5505-W/227440model.pth -d /dataset/IJB/IJB_release



170580model.pth
181952model.pth
193324model.pth
204696model.pth
216068model.pth

227440model.pth
238812model.pth
250184model.pth
261556model.pth
272928model.pth
284300model.pth
295672model.pth