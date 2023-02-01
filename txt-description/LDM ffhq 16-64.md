# FFHQ LDM 16->64
config file: models/ldm/ffhq256/ffhq16-64_sr.yaml

first stage AE: 
vq-f4 trained on 64x64 (image size 64x64)
encode vector: 3x16x16
in-dimension of Unet = 3 + 3 = 6

LDM:
downsampling factor = 4

Comments: 
Results aren't good. We decided to train 32->64 instead.

Train checkpoint & train log & img generation directory
zhren@128.32.116.228:/home/zhren/Charlie/charlie-latent-diffusion/latent-diffusion/logs/2023-01-30T21-33-48_ffhq16-64_sr/