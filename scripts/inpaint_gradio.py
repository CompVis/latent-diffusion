from omegaconf import OmegaConf
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as F
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import gradio as gr


def run(
    *,
    image,
    mask,
    device,
    model,
    sampler,
    steps,
):

    # Transpose image if needed according to EXIF data
    image = ImageOps.exif_transpose(image)

    # Save original image size
    orig_size = image.size
    print(f"Original image size: {orig_size}")

    # Convert image from PIL Image to torch tensor
    image = np.array(image.convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    # Convert mask from PIL Image to torch tensor
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    # Rescale image and mask if needed, saving unscaled original image and mask
    orig_image = image
    orig_mask = mask

    if orig_size != (512, 512):
        print("Resize image an mask to 512x512")
        image = F.resize(image, (512, 512), interpolation=F.InterpolationMode.BICUBIC)
        mask = F.resize(mask, (512, 512), interpolation=F.InterpolationMode.BICUBIC)

    # Compute the masked image
    masked_image = (1 - mask) * image

    # Saving tensors in a batch dict and move them to the GPU
    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0

    # encode masked image and concat downsampled mask
    c = model.cond_stage_model.encode(batch["masked_image"])
    cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
    c = torch.cat((c, cc), dim=1)

    # Predict image
    shape = (c.shape[1] - 1,) + c.shape[2:]
    samples_ddim, _ = sampler.sample(
        S=steps, conditioning=c, batch_size=c.shape[0], shape=shape, verbose=False
    )
    x_samples_ddim = model.decode_first_stage(samples_ddim)

    image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
    mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    # Get final image tensor by adding the original masked image with the
    # prediction inside the mask - resizing prediction image if needed
    if orig_size == (512, 512):
        inpainted = (1 - mask) * image + mask * predicted_image
        inpainted = inpainted.cpu()
    else:
        w, h = orig_size
        print(f"Resize prediction to {w}x{h}")
        predicted_image = F.resize(
            predicted_image, (h, w), interpolation=F.InterpolationMode.BICUBIC
        )
        inpainted = (1 - orig_mask) * orig_image + orig_mask * predicted_image.cpu()

    # Convert final image back to a PIL Image
    inpainted = inpainted.numpy().transpose(0, 2, 3, 1)[0] * 255
    image_result = Image.fromarray(inpainted.astype(np.uint8))

    return image_result


if __name__ == "__main__":

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    def gradio_run(sketch, nb_steps):

        image = sketch["image"]
        mask = sketch["mask"]

        generated = run(
            image=image,
            mask=mask,
            device=device,
            model=model,
            sampler=sampler,
            steps=nb_steps,
        )

        return generated

    inpaint_interface = gr.Interface(
        gradio_run,
        inputs=[
            gr.Image(interactive=True, type="pil", tool="sketch"),
            gr.Slider(minimum=1, maximum=200, value=50, label="Number of steps"),
        ],
        outputs=[
            gr.Image(),
        ],
        article="To avoid rescaling, use an image of dimensions **512x512**.",
    )

    with torch.no_grad():
        with model.ema_scope():
            inpaint_interface.launch()
