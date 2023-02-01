# FFHQ AE 64->16x16x3
config file: models/first_stage_models/vq-f4/vq-f4-64.yaml

model:
  base_learning_rate: 4.5e-06
  target: ldm.models.autoencoder.VQModel
  params:
    embed_dim: 3
    n_embed: 8192
    monitor: val/rec_loss
    ddconfig:
      double_z: false
      z_channels: 3
      resolution: 64
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult:
        - 1
        - 2
        - 4
      num_res_blocks: 2
      attn_resolutions: []
      dropout: 0.0
    lossconfig:
      target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator
      params:
        disc_conditional: false
        disc_in_channels: 3
        disc_start: 0
        disc_weight: 0.75
        codebook_weight: 1.0

Comments: encounter a bug in autoencoder that has an unantipated parameter `predicted indices`, which I refer to github issue to delete all occurances of this parameters during training. We can run it on custom datasets

Comments: reconstruction images are good

Train checkpoint & train log & img generation directory
zhren@128.32.116.228:/home/zhren/Charlie/charlie-latent-diffusion/latent-diffusion/logs/2023-01-29T22-47-12_vq-f4-64/