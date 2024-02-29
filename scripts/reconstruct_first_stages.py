import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T
import os
import torchvision.utils as vutils
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return {"model": model}, global_step


def load_and_preprocess_image(image_path, resize_shape=(256, 256)):
    transform = T.Compose([
        T.Resize(resize_shape),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def reconstruct_image(model, image_tensor):
    with torch.no_grad():
        reconstructed_img, _ = model(image_tensor)
        return reconstructed_img


def save_image(tensor, filename):
    print("Tensor Type:", type(tensor))  # Debugging line to confirm tensor type
    if isinstance(tensor, torch.Tensor):
        tensor = (tensor + 1) / 2  # Normalize if the tensor is in the range [-1, 1]
        vutils.save_image(tensor, filename)
    else:
        print("The input is not a tensor.")


def reconstruct_and_save_images(input_dir, output_dir, model):
    for image_name in os.listdir(input_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue

        image_path = os.path.join(input_dir, image_name)
        image_tensor = load_and_preprocess_image(image_path)

        reconstructed_img = reconstruct_image(model, image_tensor)

        output_path = os.path.join(output_dir, image_name)
        save_image(reconstructed_img, output_path)


def main(config_path, ckpt_path, input_dir, output_dir):
    config = OmegaConf.load(config_path)
    model_info, step = load_model_from_config(config, ckpt_path)
    model = model_info["model"]

    os.makedirs(output_dir, exist_ok=True)
    reconstruct_and_save_images(input_dir, output_dir, model)


if __name__ == "__main__":
    """
    python scripts/reconstruct_first_stages.py \
        --config ./models/first_stage_models/kl-f4/config.yaml \
        --ckpt ./models/first_stage_models/kl-f4/model.ckpt \
        --input_dir  ./eval_data \
        --output_dir ./reconstructed_images_pretrain


    python scripts/reconstruct_first_stages.py \
        --config ./logs/2024-02-24T19-56-50_autoencoder_kl_64x64x3/checkpoints/config.yaml \
        --ckpt ./logs/2024-02-24T19-56-50_autoencoder_kl_64x64x3/checkpoints/last.ckpt \
        --input_dir  ./eval_data \
        --output_dir ./reconstructed_images_train200
    """
    parser = argparse.ArgumentParser(description="Reconstruct images from training autoencoder models")
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint file')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where input images are stored')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where output images will be saved')

    args = parser.parse_args()
    main(args.config, args.ckpt, args.input_dir, args.output_dir)
