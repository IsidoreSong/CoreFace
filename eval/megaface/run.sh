#!/usr/bin/env bash

#DEVKIT="/raid5data/dplearn/megaface/devkit/experiments"
#ALGO="r100ii" #ms1mv2
#ROOT=$(dirname `which $0`)
#echo $ROOT
#python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model '../../models2/model-r100-ii/model,0'
#python -u remove_noises.py --algo "$ALGO"

#cd "$DEVKIT"
#LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH"

170580model.pth
181952model.pth
193324model.pth
204696model.pth
216068model.pth

227440model
238812model
250184model
261556model
272928model
284300model
295672model

CUDA_VISIBLE_DEVICES=1

ALGO="272928model"
python -u /workspace/_evaluation_/megaface/gen_megaface.py --gpu 0 --algo "$ALGO" \
  --model /dataset/model/ArcFace_Emore_d46_5505/$ALGO.pth


python -u /workspace/_evaluation_/megaface/remove_noises.py --algo "$ALGO"

python -u /workspace/_evaluation_/megaface/devkit/experiments/run_experiment.py "/dataset/megaface/feature_out_clean/megaface" \
  "/dataset/megaface/feature_out_clean/facescrub" _"$ALGO".bin \
  /dataset/megaface/result/$ALGO/ -s 1000000 \
  -p /dataset/megaface/devkit/templatelists/facescrub_features_list.json \
  -dlp /dataset/megaface/devkit/templatelists

ls /dataset/megaface/result/$ALGO/cmc_facescrub_megaface_238812model_1000000_1.json

cd -

