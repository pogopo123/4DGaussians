#!/usr/bin/env bash
# Bottleneck probe: baseline vs fat-MLP vs hi-res-HexPlane tren cung 1 scene, cung seed (6666).
set -u
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Gaussians4D

SCENE=backpack_frame0_v2
SRC=data/multipleview/$SCENE
TESTIT="1000 3000 5000 7000 10000 12000 14000 15000"
SAVEIT="14000 15000"

run () {  # run <tag> <gpu> <port>
  local tag=$1 gpu=$2 port=$3
  mkdir -p output/multipleview/$tag
  CUDA_VISIBLE_DEVICES=$gpu python train.py \
      -s $SRC --port $port \
      --expname "multipleview/$tag" \
      --configs customer_scripts/$tag.py \
      --test_iterations $TESTIT \
      --save_iterations $SAVEIT \
      > output/multipleview/$tag/run.log 2>&1
  echo "[$tag] exit=$? $(date)" >> output/multipleview/probe_status.txt
}

run bn_baseline "$1" 6031 &
run bn_fatmlp   "$2" 6032 &
run bn_hires    "$3" 6033 &
wait
echo "ALL DONE $(date)" >> output/multipleview/probe_status.txt
