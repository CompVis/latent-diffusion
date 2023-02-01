# FFHQ LDM 16->256
config file: models/ldm/ffhq256/ffhq16-256_sr.yaml

first stage AE: 
vq-f16 (input image = 256x256)
encode vector: 8x16x16
in-dimension of Unet = 8 + 3 = 11

LDM:
downsampling factor = 16
Used as baseline to compare results

Comments: 
We should train on 32->256 as well.

Train checkpoint & train log & img generation directory
zhren@128.32.116.228:/home/zhren/Charlie/charlie-latent-diffusion/latent-diffusion/logs/2023-01-26T12-28-17_ffhq16-256_sr/