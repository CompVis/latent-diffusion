"""

BOLD5000 dataset folder path looks like

    ├── CSI1
    │  ├── BOLD5000_CSI1_sess1
    │    ├──Behavioral_Data
    │    ├──BOLD_Raw
    │      ├──01_BOLD_CSI1_Sess-13_Run-1
    │         ├── concat_file
    │          ├── conc.03-0001-000001.npy
    │          ├── conc.03-0001-000002.npy
    │          ├── ...
    │        ├── ...
    │      ├──02_BOLD_CSI1_Sess-13_Run-2
    │      ├──03_BOLD_CSI1_Sess-13_Run-3
    │      ├── ...
    │    ├──physio
    │    ├──DICOM_log_171214112502.txt
    │  ├── BOLD5000_CSI1_sess2
    │   ├── ...

    ├── CSI2
    │   ├── ...
    
    Label file name changed Scene to Scenes
    Presented Stimulus files looks like
    ├── COCO
    │   ├── COCO_train2014_000000000036.jpg
    │   ├── COCO_train2014_000000000584.jpg
    │   ├── ...
    ├── ImageNet
    │   ├── n01440764_10110.JPEG
    │   ├── n01440764_13744.JPEG
    │   ├── ...
    ├── Scenes
    │   ├── ...

"""
import numpy as np
import cv2
from torch.utils.data import Dataset
from natsort import natsorted
import glob
import itertools
import random
import math
import os

def resize_image(image,dim):
    image = cv2.resize(image,(dim,dim))
    return image

def convert_rgb(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def filter_list(directory,list_for):
    directory = list(filter(lambda item: list_for in item, directory))
    return directory
def flat_paths(directory):
    directory = list(itertools.chain(*directory))
    return directory


class MriActivityBase(Dataset):
    def __init__(self, data_root,stim_root,subjects,subDataset, img_dim):
        self.parent_dir = data_root
        self.stim_dir = stim_root
        self.subjects = subjects
        self.subDataset = subDataset
        self.img_dim = img_dim
        self.activity_file = 'concat_file'
        self.behavioral_Data = 'Behavioral_Data'
        self.data = dict()

        if not self.img_dim:
            self.img_dim = 64
################################################################################
        stim_path = self.vision_label(stim_root)
        self.parent_dir = sorted(flat_paths([glob.glob(data_root + subject + "/*") for subject in subjects]))
        behavioral_dir = self.parent_dir[slice(len(self.parent_dir))]
        for item in range(len(self.parent_dir)):
            self.parent_dir[item] = glob.glob(os.path.join(self.parent_dir[item], "BOLD_Raw") + "/*")
            behavioral_dir[item] = glob.glob(os.path.join(behavioral_dir[item], self.behavioral_Data) + "/*")
            behavioral_dir[item] = filter_list(behavioral_dir[item], 'tsv')
        for x in range(len(self.parent_dir)):
            for j in range(len(self.parent_dir[x])):
              self.parent_dir[x][j] = os.path.join(self.parent_dir[x][j], self.activity_file)
        self.parent_dir = flat_paths(self.parent_dir)
        self.parent_dir = filter_list(self.parent_dir, 'Run')
        for activity in self.parent_dir:
            self.parent_dir[self.parent_dir.index(activity)] = glob.glob(activity + "/*")
        scene = list()
        for data_paths in behavioral_dir:
            data_paths = natsorted(data_paths)
            for path in data_paths:
                if os.stat(path).st_size == 0:
                    raise Exception(path + ' is damaged')
                with open(path, "r") as f:
                    file = f.read().splitlines()
                    for item in file: file[file.index(item)] = item.split("\t")
                    img_name_idx = file[0].index("ImgName")
                    img_type_idx = file[0].index("ImgType")
                    for item in file[1:]:
                        for key in stim_path:
                            if key.lower() in item[img_type_idx]:
                                scene.append(os.path.join(stim_path[key], item[img_name_idx]))
        self.data["file_path_"] = scene
        self.data["relative_file_path_"] = [run[x:x+5] for run in [ses[3:188] for ses in self.parent_dir] for x in range(0, len(run),5)]
        
        if subDataset is not None:
            delete_indices = []
            subDataset = subDataset.lower()
            for index, item in enumerate(self.data["file_path_"]):
                if subDataset not in item.lower():
                    delete_indices.append(index)
            delete_indices.sort(reverse=True)
            for index in delete_indices:
                for key in self.data:
                    del self.data[key][index]
        
        if self.subDataset is not None:
            delete_indices = []
            self.subDataset = self.subDataset.lower()
            for index, item in enumerate(self.data["file_path_"]):
                if subDataset not in item.lower():
                    delete_indices.append(index)
            delete_indices.sort(reverse=True)
            for index in delete_indices:
                for key in self.data:
                    del self.data[key][index]
        
        self.data["relative_file_path_"], self.data["file_path_"] = self.getNamefor(self)

        combined = list(zip(self.data["file_path_"] , self.data["relative_file_path_"]))
        random.seed(1234)
        random.shuffle(combined)
        self.data["file_path_"], self.data["relative_file_path_"] = zip(*combined)
        # start fixation cross - 6 secs
        # end fixation cross -12 secs
        # acqusition time - 2 secs
        # hemodynamic response about 5 slices (10 secs)
        # Eliminate start and end fixation cross in each run
        
    
    def __len__(self):
        return len(self.data["relative_file_path_"])
    
    def getNamefor(self, data):
        start = 0
        end = len(self.data["relative_file_path_"])
        step = end // 5
        
        if "Train" in type(self).__name__:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][start:start+step*3]
            self.data["file_path_"] = self.data["file_path_"][start:start+step*3]

        elif "Valid" in type(self).__name__:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][start+step*3:start+step*4]
            self.data["file_path_"] = self.data["file_path_"][start+step*3:start+step*4]

        else:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][start+step*4:end]
            self.data["file_path_"] = self.data["file_path_"][start+step*4:end]
    
        return self.data["relative_file_path_"], self.data["file_path_"]
    
    def _vision_label(self,stim_root):
        stim_type = glob.glob(stim_root + "*")
        stim_path = {}
        for item in stim_type:
            stim_path[item.split("\\")[1]] = item
        return stim_path
    
    def vision_label(self, stim_root):
        import string
        alphabet = list(string.ascii_letters)
        stim_type = glob.glob(stim_root + "*")
        stim_path = {}
        for item in stim_type:
            stim_path[item.split(stim_root[:-1])[1]] = item
        new_dict = {}
        for key, value in stim_path.items():
            for char in key:
                if char not in alphabet:
                    new_key = key.replace(char,"")
            new_dict[new_key] = value
        return new_dict
    
    def __getitem__(self, i):    
        #image = {key : cv2.resize(np.load(key,allow_pickle=True),(512,512)) for key in self.data["file_path_"][i]}  
        image = {key : np.load(key,allow_pickle=True) for key in self.data["relative_file_path_"][i]}  
        path_label = self.data["file_path_"][i]
        stimulus = cv2.imread(path_label)
        stimulus = convert_rgb(stimulus)
        stimulus = resize_image(stimulus,self.img_dim)
        example = {
            "image" : stimulus,
            "activity" : np.array(list(image.values())).transpose(1,2,0),
            "relative_file_path_" : list(image.keys()),
            #"paths_hemodynamic_response" : list(image.keys()),
            "file_path_" : self.data["file_path_"][i]
            #"path_label" : path_label 
            }
        
        return example
                                
class activityTrain(MriActivityBase):
    def __init__(self, **kwargs):
        super().__init__(data_root = '/content/drive/MyDrive/BOLD5000/',stim_root = '/content/drive/MyDrive/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/',subjects = ["CSI1","CSI2","CSI3","CSI4"],subDataset="ImageNet", img_dim=None)

class activityValidation(MriActivityBase):
    def __init__(self, **kwargs):
        super().__init__(data_root = '/content/drive/MyDrive/BOLD5000/',stim_root = '/content/drive/MyDrive/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/',subjects = ["CSI1","CSI2","CSI3","CSI4"],subDataset="ImageNet", img_dim=None)

class activityTest(MriActivityBase):
    def __init__(self, **kwargs):
        super().__init__(data_root = '/content/drive/MyDrive/BOLD5000/',stim_root = '/content/drive/MyDrive/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/',subjects = ["CSI1","CSI2","CSI3","CSI4"],subDataset="ImageNet", img_dim=None)

