import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
import lpips
import clip

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from pytorch_wavelets import DWTForward
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from omegaconf import OmegaConf

from ldm.models.autoencoder_wav import AutoencoderKL


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(args.gpu)
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}')
    dist.init_process_group(
        backend='nccl', 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank,
    )


def is_main_process():
    return dist.get_rank() == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE Analysis', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--data_path', default='/path/to/imagenet-1k', type=str)
    parser.add_argument('--resos', default=256, type=int)
    
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    
    # multi-node and multi-GPU evaluation 
    init_distributed_mode(args)

    # data loading
    transform = transforms.Compose([
        transforms.Resize(args.resos),
        transforms.CenterCrop((args.resos, args.resos)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=transform)  
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    image_val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # build the model
    config = OmegaConf.load('configs/autoencoder/autoencoder_kl-wav_16x16x16.yaml')
    vae = AutoencoderKL.load_from_checkpoint(
        'logs/ch1-2-4_baseline/last.ckpt',
        ddconfig=config.model.params.ddconfig,
        lossconfig=config.model.params.lossconfig,
        embed_dim=config.model.params.embed_dim,
    )
    vae = vae.to(args.device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')
    
    # evaluate the reconstruction loss [-1, 1] range 
    total_loss = {key: 0.0 for key in ['reconstruction', 'low_frequency', 'high_frequency', 'perceptual', 'clip_semantic', 'sam_semantic']}
    total_images = 0
    batch_imgs_to_save = 10
    visualize_imgs = [] if is_main_process() else None
    
    # for evaluation metrics
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(args.device)
    l2_loss = nn.MSELoss(reduction='mean')
    
    lpips_loss = lpips.LPIPS(net='vgg').to(args.device).eval()
    
    clip, preprocess_clip = clip.load('ViT-B/32', device=args.device)
    clip.eval()
    
    sam = sam_model_registry['vit_b'](checkpoint='/path/to/sam_vit_b_01ec64.pth')
    sam = sam.to(args.device).eval()
    sam_predictor = SamPredictor(sam)

    with torch.no_grad():
        for imgs, labels in tqdm(image_val_loader, disable=not is_main_process(), desc='Processing images', leave=True):
            imgs = imgs.to(args.device)
            rec_imgs, _, = vae(imgs)
            
            # first level DWT
            ll1, hs = dwt(imgs)
            lh1, hl1, hh1 = hs[0][:, 0], hs[0][:, 1], hs[0][:, 2]
            rec_ll1, rec_hs = dwt(rec_imgs)
            rec_lh1, rec_hl1, rec_hh1 = rec_hs[0][:, 0], rec_hs[0][:, 1], rec_hs[0][:, 2]
            
            # preprocess for CLIP, which expects input of size (224, 224)
            # more efficient than applying preprocess_clip() sample-by-sample, but introduce slight discrepancies 
            # due to differences between PIL and tensor-based Resize implementations in torchvision
            imgs_clip = torch.clamp((imgs + 1) / 2, min=0, max=1)
            imgs_clip = F.resize(imgs_clip, 224, F.InterpolationMode.BICUBIC)
            imgs_clip = F.normalize(imgs_clip, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            rec_imgs_clip = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
            rec_imgs_clip = F.resize(rec_imgs_clip, 224, F.InterpolationMode.BICUBIC)
            rec_imgs_clip = F.normalize(rec_imgs_clip, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

            # imgs_clip = torch.clamp((imgs + 1) / 2, min=0, max=1).cpu()
            # imgs_clip = torch.stack([preprocess_clip(torchvision.transforms.ToPILImage()(img)) for img in imgs_clip]).to(args.device)
            # rec_imgs_clip = torch.clamp((rec_imgs + 1) / 2, min=0, max=1).cpu()
            # rec_imgs_clip = torch.stack([preprocess_clip(torchvision.transforms.ToPILImage()(img)) for img in rec_imgs_clip]).to(args.device)
            
            features_clip = clip.encode_image(imgs_clip)
            rec_features_clip = clip.encode_image(rec_imgs_clip)

            # preprocess for SAM, which expects input of size (1024, 1024)
            # slightly differs from apply_image(), which uses uint8 NumPy arrays
            imgs_sam = torch.clamp((imgs + 1) / 2, min=0, max=1).mul_(255)
            imgs_sam = sam_predictor.transform.apply_image_torch(imgs_sam)
            rec_imgs_sam = torch.clamp((rec_imgs + 1) / 2, min=0, max=1).mul_(255)
            rec_imgs_sam = sam_predictor.transform.apply_image_torch(rec_imgs_sam)
            
            sam_predictor.set_torch_image(imgs_sam, (256, 256))
            features_sam = sam_predictor.features
            features_sam = features_sam.reshape(imgs.shape[0], -1)
            sam_predictor.set_torch_image(rec_imgs_sam, (256, 256))
            rec_features_sam = sam_predictor.features
            rec_features_sam = rec_features_sam.reshape(imgs.shape[0], -1)

            batch_losses = {
                'reconstruction': l2_loss(rec_imgs, imgs).item(),
                'low_frequency': l2_loss(rec_ll1, ll1).item(),
                'high_frequency': (l2_loss(rec_lh1, lh1).item() + l2_loss(rec_hl1, hl1).item() + l2_loss(rec_hh1, hh1).item()) / 3,
                'perceptual': lpips_loss(rec_imgs, imgs).mean().item(),
                'clip_semantic': 1 - nn.functional.cosine_similarity(features_clip, rec_features_clip).mean().item(),
                'sam_semantic': 1 - nn.functional.cosine_similarity(features_sam, rec_features_sam).mean().item(),
            }
            
            for key in total_loss:
                total_loss[key] += batch_losses[key] * imgs.shape[0]

            total_images += imgs.shape[0]
            
            if is_main_process() and len(visualize_imgs) < batch_imgs_to_save:
                visualize_imgs.append(rec_imgs[:4].cpu())
    
    # aggregate losses across all distributed processes
    for key in total_loss:
        t = torch.tensor(total_loss[key], device=args.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_loss[key] = t.item()

    t = torch.tensor(total_images, device=args.device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    total_images = t.item()

    if is_main_process():
        save_dir = 'outputs/analysis_figures'
        os.makedirs(save_dir, exist_ok=True)

        # visualize some reconstructed images
        visualize_imgs = torch.cat(visualize_imgs, dim=0)
        visualize_imgs = torch.clamp((visualize_imgs + 1) / 2, min=0, max=1)
        visualize_imgs = torchvision.utils.make_grid(visualize_imgs, nrow=8, padding=0)
        visualize_imgs = visualize_imgs.permute(1, 2, 0).mul_(255).numpy()
        visualize_imgs = Image.fromarray(visualize_imgs.astype(np.uint8))
        visualize_imgs.save(f'{save_dir}/recon_kl-vae-wav-f16c16.png')
        
        # compute average loss per component
        avg_loss = {key: total_loss[key] / total_images for key in total_loss}
        
        # save results
        csv_path = f'{save_dir}/loss_metrics.csv'

        new_row = {
            'Model': 'KL-VAE-WAV-f16c16',
            'Dataset': 'ImageNet',
            'Reconstruction Loss': avg_loss['reconstruction'],
            'Low Frequency Loss': avg_loss['low_frequency'],
            'High Frequency Loss': avg_loss['high_frequency'],
            'Perceptual Loss': avg_loss['perceptual'],
            'CLIP Semantic Loss': avg_loss['clip_semantic'],
            'SAM Semantic Loss': avg_loss['sam_semantic'],
        }

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(csv_path, index=False)
