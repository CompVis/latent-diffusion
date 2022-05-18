


try:
    import torch
    import torch.nn as nn
except:
    pass

try:
    import torchvision as tv
    import torchvision.transforms as TT
    import torchvision.transforms.functional as TF
except:
    pass

try:
    import torchmetrics
except:
    pass

try:
    import kornia
except:
    pass

try:
    import cupy
except:
    pass

try:
    import pytorch_lightning as pl
except:
    pass

try:
    import detectron2
    from detectron2 import model_zoo as _
    from detectron2 import engine as _
    from detectron2 import config as _
    from detectron2 import data as _
    from detectron2.utils import visualizer as _
except:
    pass


#################### UTILITIES ####################

# @cupy.memoize(for_each_device=True)
def cupy_launch(func, kernel):
    return cupy.cuda.compile_with_cache(kernel).get_function(func)

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

def channel_squeeze(x, dim=1):
    a = x.shape[:dim]
    b = x.shape[dim+2:]
    return x.reshape(*a, -1, *b)
def channel_unsqueeze(x, shape, dim=1):
    a = x.shape[:dim]
    b = x.shape[dim+1:]
    return x.reshape(*a, *shape, *b)




