#!/bin/bash

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python main.py --no-test --base configs/latent-diffusion/danbooru-keypoints-ldm-vq-4.yaml -t --gpus 0,
