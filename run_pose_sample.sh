#!/bin/bash
for i in `ls example_pose_keypoints`;
do
    python scripts/sample_pose_diffusion.py -r models/ldm/keypoints_danbooru256/model.ckpt -p example_pose_keypoints/$i -l /tmp/logdir -n 50 --batch_size 1 -c 500 -e 0.0
done
