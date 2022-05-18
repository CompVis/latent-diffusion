import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torch
from torch import LongTensor, Tensor
import json


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            #self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            #self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
            self.preprocessor = albumentations.Compose([self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def tokenize_coordinates(self, x: float, y: float, dim:int) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        #x_discrete = int(round(x * (self.512 - 1)))
        #y_discrete = int(round(y * (self.512 - 1)))
        if x > (dim - 1):
            x = (dim - 1)
        if y > (dim - 1):
            y = (dim - 1)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        return int(round(x)) + int(round(y) * dim)

    def coordinates_from_token(self, token: int, dim: int) -> (float, float):
        x = token % dim
        y = token // dim
        return x, y

    def build_tensor_from_kps(self, kps, dim):
        kps_names = ['nose',
                     'eye_left',
                     'eye_right',
                     'ear_left',
                     'ear_right',
                     'shoulder_left',
                     'shoulder_right',
                     'elbow_left',
                     'elbow_right',
                     'wrist_left',
                     'wrist_right',
                     'hip_left',
                     'hip_right',
                     'knee_left',
                     'knee_right',
                     'ankle_left',
                     'ankle_right']
        tokens = []
        for name in kps_names:
           x = kps[name][0]
           y = kps[name][1]
           if dim != 512:
               x = x // (512/dim)
               y = y // (512/dim)
           _token = self.tokenize_coordinates(x, y, dim)
           tokens.append(_token)
        #return LongTensor(tokens)
        return Tensor(tokens)

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        keypoints_json_file_path = self.labels["file_path_"][i][:-4].replace('images', 'keypoints') + '.json'
        example["keypoints"] = self.build_tensor_from_kps(json.loads(open(keypoints_json_file_path, 'r').read()), dim=32)
        #example["keypoints"] = torch.ones([17])
        example["keypoints"] = torch.unsqueeze(example["keypoints"], 0)
        #print("dataset keypoint shape:")
        #print(example["keypoints"].shape)
        #print(example["keypoints"])
        #print("dataset keypoint shape:-----end")
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
