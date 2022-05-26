import argparse
import os
import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
from time import time
from torch import device, Tensor
from typing import List, Tuple, Any, Dict, Callable

from PIL import Image

from ldm.util import instantiate_from_config, ismap
from ldm.models.diffusion.ddim import DDIMSampler


def get_cond(path: Path, device: device):
    up_f = 4
    c = Image.open(path)
    c = torch.unsqueeze(transforms.ToTensor()(c), 0)
    c_up = transforms.functional.resize(
        c,
        size=[up_f * c.shape[2], up_f * c.shape[3]],
        antialias=True)
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')
    c = rearrange(c, '1 c h w -> 1 h w c')
    c = 2. * c - 1.
    c = c.to(device)
    example = {
        "LR_image": c,
        "image": c_up,
    }
    return example


@torch.no_grad()
def convsample_ddim(
    model: nn.Module,
    cond: Tensor,
    steps: int,
    shape: Tuple,
    eta: float = 1.0,
    callback: Callable = None,
    normals_sequence: Any = None,
    mask: Tensor = None,
    x0: Tensor = None,
    quantize_x0: bool = False,
    temperature: float = 1.,
    score_corrector: Any = None,
    corrector_kwargs: Dict = None,
    x_T: Tensor = None,
    **kwargs,
):
    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(
        steps,
        batch_size=bs,
        shape=shape,
        conditioning=cond,
        callback=callback,
        normals_sequence=normals_sequence,
        quantize_x0=quantize_x0,
        eta=eta,
        mask=mask,
        x0=x0,
        temperature=temperature,
        verbose=False,
        score_corrector=score_corrector,
        corrector_kwargs=corrector_kwargs,
        x_T=x_T)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(
    batch: Tensor,
    model: nn.Module,
    mode: str = "vanilla",
    custom_steps: int = None,
    eta: float = 1.0,
    swap_mode: bool = False,
    masked: bool = False,
    quantize_x0: bool = False,
    custom_shape: Tuple = None,
    temperature: float = 1.,
    noise_dropout: float = 0.,
    corrector: Any = None,
    corrector_kwargs: Dict = None,
    x_T: Tensor = None,
    save_intermediate_vid: bool = False,
    ddim_use_x0_pred: bool = False,
):
    z, c, x, xrec, xc = model.get_input(
        batch,
        model.first_stage_key,
        return_first_stage_outputs=True,
        force_c_encode=not (hasattr(model, 'split_input_params') and model.cond_stage_key == 'coordinates_bbox'),
        return_original_cond=True)

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log = dict()
    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)
    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0, img_cb = time(), None
        sample, intermediates = convsample_ddim(
            model,
            c,
            steps=custom_steps,
            shape=z.shape,
            eta=eta,
            quantize_x0=quantize_x0,
            img_callback=img_cb,
            mask=None,
            x0=z0,
            temperature=temperature,
            noise_dropout=noise_dropout,
            score_corrector=corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t)
        t1 = time()
        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)
    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0
    return log


def proc_image(logs: Dict):
    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    return Image.fromarray(sample[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing images to apply super resolution"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    config = OmegaConf.load("models/ldm/bsr_sr/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load("models/ldm/bsr_sr/model.ckpt")["state_dict"],
        strict=False)

    dvc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(dvc)

    os.makedirs(opt.outdir, exist_ok=True)
    img_paths = list(Path(opt.indir).glob("*"))
    with torch.no_grad():
        for img_path in img_paths:
            example = get_cond(img_path, dvc)
            height, width = example["image"].shape[1:3]
            split_input = height >= 128 or width >= 128
            if split_input:
                ks, stride, vqf = 128, 64, 4
                model.split_input_params = {
                    "ks": (ks, ks),
                    "stride": (stride, stride),
                    "vqf": vqf,
                    "patch_distributed_vq": True,
                    "tie_braker": False,
                    "clip_max_weight": 0.5,
                    "clip_min_weight": 0.01,
                    "clip_max_tie_weight": 0.5,
                    "clip_min_tie_weight": 0.01,
                }
            else:
                if hasattr(model, "split_input_params"):
                    delattr(model, "split_input_params")
            logs = make_convolutional_sample(
                example,
                model,
                mode="ddim",
                custom_steps=100,
                eta=1.0,
                swap_mode=False,
                masked=False,
                quantize_x0=False,
                custom_shape=None,
                temperature=1.,
                noise_dropout=0.,
                corrector=None,
                corrector_kwargs=None,
                x_T=None,
                save_intermediate_vid=False,
                ddim_use_x0_pred=False,
            )
            img = proc_image(logs)
            img.save(Path(opt.outdir) / img_path.name)