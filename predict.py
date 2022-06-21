import time
import typing
import uuid
import os

import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import trange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


class Predictor(BasePredictor):
    def setup(self):
        start_time = time.time()
        print(f'Performing setup!')

        config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
        print(f'Model loaded at {time.time() - start_time}')

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        print(f'Model loaded on device at {time.time() - start_time}')

        sampler = PLMSSampler(model)
        print(f'Sampler loaded at {time.time() - start_time}')

        self.model = model
        self.sampler = sampler

        print(f'Setup complete at {time.time() - start_time}')

    def predict(
            self,
            prompt: str = Input(description="Image prompt"),
            scale: float = Input(description="Unconditional guidance, increase for improved quality and less diversity",
                                 default=5.0),
            steps: int = Input(description="Number of diffusion steps", default=50),
            eta: float = Input(description="ddim_eta (recommend leaving at default of 0 for faster sampling)",
                               default=0),
            plms: bool = Input(description="Sampling method requiring fewer steps (e.g. 25) to get good quality images",
                               default=True),
            batch_size: int = Input(description="Number of images to generate per batch", default=4),
            batches: int = Input(description="Number of batches", default=1),
            width: int = Input(description="Width of images (use a multiple of 8 e.g. 256)", default=256),
            height: int = Input(description="Height of images (use a multiple of 8 e.g. 256)", default=256)
    ) -> typing.List[Path]:

        print(f'Prediction started!')

        if plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)
        print(f'Sampler loaded ')

        all_samples = list()
        with torch.no_grad():
            with self.model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = self.model.get_learned_conditioning(batch_size * [""])
                for n in trange(batches, desc="Sampling"):
                    c = self.model.get_learned_conditioning(batch_size * [prompt])
                    shape = [4, height // 8, width // 8]
                    samples_ddim, _ = self.sampler.sample(S=steps,
                                                          conditioning=c,
                                                          batch_size=batch_size,
                                                          shape=shape,
                                                          verbose=False,
                                                          unconditional_guidance_scale=scale,
                                                          unconditional_conditioning=uc,
                                                          eta=eta)

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image_path = f'{uuid.uuid4()}.png'
                        Image.fromarray(x_sample.astype(np.uint8)).save(image_path)
                        yield Path(image_path)
                    all_samples.append(Path(image_path))
                return all_samples
