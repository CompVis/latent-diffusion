import argparse
import datetime
import os
import torch
import yaml

import numpy as np

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pathlib import Path
from torch import Tensor

from PIL import Image

from scripts.sample_diffusion import load_model

rescale = lambda x: (x + 1.) / 2.


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=Path,
        required=True,
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=10
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=Path,
        required=True,
        help="extra logdir",
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    return parser


def custom_to_pil(x: Tensor):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


@torch.no_grad()
def run():
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    opt, _ = parser.parse_known_args()
    ckpt = opt.resume if opt.resume.is_file() else opt.resume / 'model.ckpt'
    logdir = opt.logdir

    config = OmegaConf.load(ckpt.parent / 'config.yaml')
    print(config)

    model, global_step = load_model(config, ckpt, gpu=True, eval_mode=True)
    gss = f'{global_step:02d}'
    logdir = logdir / 'samples' / gss / now
    os.makedirs(logdir, exist_ok=True)
    print(f"global step: {global_step}")
    print(75 * "=")
    print(f"logging to: {logdir}")

    sampling_file = logdir / 'sampling_config.yaml'
    sampling_conf = vars(opt)
    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    assert model.cond_stage_key == 'class_label'
    ddim = DDIMSampler(model)
    bs = opt.n_samples
    eta = opt.eta
    steps = opt.custom_steps
    shape = [
        model.model.diffusion_model.in_channels,
        model.model.diffusion_model.image_size,
        model.model.diffusion_model.image_size,
    ]
    with model.ema_scope("Plotting"):
        batch = {
            'class_label': torch.full((bs,), 1).cuda(),
        }
        c = model.cond_stage_model(batch)
        samples, intermediates = ddim.sample(
            S=steps,
            batch_size=bs,
            conditioning=c,
            shape=shape,
            eta=eta,
        )
        x_samples = model.decode_first_stage(samples)
        for i in range(bs):
            img = x_samples[i]
            img = custom_to_pil(img)
            img_path = logdir / 'img' / f'sample_{i:06}.png'
            os.makedirs(img_path.parent, exist_ok=True)
            img.save(img_path)
 


if __name__ == '__main__':
    run()