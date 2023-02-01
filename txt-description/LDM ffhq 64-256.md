# FFHQ LDM 64->256
config file: models/ldm/ffhq256/ffhq256_sr.yaml

first stage AE: 
vq-f4 (input image = 256x256)
encode vector: 3x64x64
in-dimension of Unet = 3 + 3 = 6

LDM:
downsampling factor = 4
Used as second cascaded model

Train checkpoint & train log & img generation directory
zhren@128.32.116.228:/home/zhren/Charlie/charlie-latent-diffusion/latent-diffusion/logs/2023-01-21T17-53-30_ffhq256_sr