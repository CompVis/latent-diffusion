# Doing inference code on server


# Read the image_path from folder
import os
ffhq_imgs = os.listdir(".\\data\\ffhq\\test_img\\")
print(ffhq_imgs)

# baseline
from notebook_helpers import get_local_model


from notebook_helpers import run
import os
import torch
import numpy as np
# import IPython.display as d
from PIL import Image
from datetime import datetime

psnrs_BL = np.array([])
ssims_BL = np.array([])
import skimage
import os
import cv2
import glob
import numpy as np
test_dir = ".\\data\\ffhq\\generated_img\\original256\\"
baseline_dir = f".\\data\\ffhq\\generated_img\\server\\BL_epoch_268\\"
FS_dir = ".\\data\\ffhq\\generated_img\\FS_216\\"
cascade_dir = ".\\data\\ffhq\\generated_img\\SS_FS_109\\"

# Helper methods
def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')
def sort_by_number_ending(lst): # credit chatGPT
    return sorted(lst, key=lambda x: int(x[:-4].split("-")[-1]))

mode = "FS"

if mode == "BL":
    print("Selected baseline")
    path_conf = ".\\models\\trained_model_config\\ffhq32-256_sr.yaml"
    path_ckpt = ".\\trained_models\\epoch=000268_ldm-32-256.ckpt"
    model_bl = get_local_model(path_conf, path_ckpt) # load model (bl:=baseline)
    up_f_bl = 8
    store_dir = baseline_dir
    custom_steps = 200
    if not os.path.exists(store_dir):
            os.makedirs(store_dir)
    for idx, img in enumerate(ffhq_imgs):
        cond_choice_path = os.path.join(".\\data\\ffhq\\test_img\\", img)
        logs = run(model_bl["model"], cond_choice_path, "superresolution", custom_steps, up_f_bl, img_idx=idx)
        # 拿到了BL的结果，存起来
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        print(sample.shape)
        a = Image.fromarray(sample[0]) # it's now an image
        a.save(f"{baseline_dir}BL32-256-{idx}.png") # 我们这边需要存起来因为FID是直接在两个数据集上work的
elif mode == "FS":
    print("Selected FS")
    path_conf = ".\\models\\trained_model_config\\ffhq32-64_sr.yaml"
    path_ckpt = ".\\trained_models\\epoch=000067_ldm-32-64.ckpt"
    model_lr = get_local_model(path_conf, path_ckpt) # load model
    up_f_lr = 2
    FS_dir = "FS_216"
    custom_steps = 200
    if not os.path.exists(f".\\data\\ffhq\\generated_img\\{FS_dir}\\"):
        os.makedirs(f".\\data\\ffhq\\generated_img\\{FS_dir}\\")
    for idx, img in enumerate(os.listdir(".\\data\\ffhq\\test_img\\")):
        cond_choice_path = os.path.join(".\\data\\ffhq\\test_img\\", img)
        logs = run(model_lr["model"], cond_choice_path, "superresolution", custom_steps, up_f_lr)
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        print(sample.shape)
        a = Image.fromarray(sample[0]) # it's now an image
        a.save(f".\\data\\ffhq\\generated_img\\{FS_dir}\\FS32-64-{idx}.png")
elif mode == "SS":
    print("Selected SS")
    path_conf = ".\\models\\trained_model_config\\ffhq256_sr.yaml"
    path_ckpt = ".\\trained_models\\epoch=000052_ldm-64-256.ckpt"
    model_hr = get_local_model(path_conf, path_ckpt) # load model
    up_f_hr = 4 
    def sort_by_number_ending(lst): # credit chatGPT
        return sorted(lst, key=lambda x: int(x[:-4].split("-")[-1]))
    custom_steps = 200
    if not os.path.exists(f".\\data\\ffhq\\generated_img\\SS_{FS_dir}\\"):
        os.makedirs(f".\\data\\ffhq\\generated_img\\SS_{FS_dir}\\")
    for idx, img in enumerate(sort_by_number_ending(os.listdir(f".\\data\\ffhq\\generated_img\\{FS_dir}"))):
        cond_choice_path = os.path.join(f".\\data\\ffhq\\generated_img\\{FS_dir}", img)
        print(cond_choice_path)
        logs = run(model_hr["model"], cond_choice_path, "superresolution", custom_steps, up_f_hr, downsample = False)
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        print(sample.shape)
        a = Image.fromarray(sample[0]) # it's now an image
        a.save(f".\\data\\ffhq\\generated_img\\SS_{FS_dir}\\SS64-256-{idx}.png")

# original_imgs = sort_by_number_ending(glob.glob(os.path.join(test_dir, "origin*.png")))
# baseline_imgs = sort_by_number_ending(os.listdir(baseline_dir)) # BL: Basline (32->256)
# cascade_imgs = sort_by_number_ending(os.listdir(cascade_dir)) # SS: Second stage

# for idx, img in enumerate(original_imgs):
#     input_img_path = img
#     test_img_path = os.path.join(baseline_dir, baseline_imgs[idx])
#     input_img = cv2.imread(input_img_path)
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     test_img = cv2.imread(test_img_path)
#     test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#     psnr = skimage.metrics.peak_signal_noise_ratio(input_img, test_img, data_range=255)
#     psnrs_BL = np.append(psnrs_BL, psnr)
#     ssim = skimage.metrics.structural_similarity(input_img, test_img, data_range=255, multichannel=True)
#     ssims_BL = np.append(ssims_BL, ssim)

# import matplotlib.pyplot as plt
# import numpy as np

# # generate data for histogram
# # data = np.random.normal(100, 20, 1000)

# # create the subplots
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# # plot the first histogram
# ax[0].hist(psnrs_BL, bins=30, edgecolor='black')
# ax[0].set_xlabel('PSNR')
# ax[0].set_ylabel('Frequency')
# ax[0].set_title('Original 32->256 PSNR epoch 195')

# # plot the second histogram
# ax[1].hist(ssims_BL, bins=30, edgecolor='black')
# ax[1].set_xlabel('SSIM')
# ax[1].set_ylabel('Frequency')
# ax[1].set_title('Original 32->256 SSIM epoch 195')

# # adjust spacing between subplots
# fig.tight_layout()

# # display plot
# plt.show()
# print("Average of psnrs in Original 32->256 is", np.average(psnrs_BL))
# print("Stadard Deviation of psnrs in Original 32->256 is", np.std(psnrs_BL))
# print("Average of ssims in Original 32->256 is", np.average(ssims_BL))
# print("Stadard Deviation of ssims in Original 32->256 is", np.std(ssims_BL))
