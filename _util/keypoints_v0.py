


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d


######################## COCO SETUP ########################

# visibility:
#     v=0: not labeled (in which case x=y=0)
#     v=1: labeled but not visible
#     v=2: labeled and visible  
coco_keypoints = [
    'nose',             #  0
    'eye_left',         #  1
    'eye_right',        #  2
    'ear_left',         #  3
    'ear_right',        #  4

    'shoulder_left',    #  5
    'shoulder_right',   #  6
    'elbow_left',       #  7
    'elbow_right',      #  8
    'wrist_left',       #  9
    'wrist_right',      # 10

    'hip_left',         # 11
    'hip_right',        # 12
    'knee_left',        # 13
    'knee_right',       # 14
    'ankle_left',       # 15
    'ankle_right',      # 16
]
coco_keypoints_ext = coco_keypoints + [
    'nose_root',    # 17
    'body_upper',   # 18
    'thumb_left',   # 19
    'thumb_right',  # 20
    'toe_left',     # 21
    'toe_right',    # 22
]
coco_annotation_order = [
    4,2,0,1,3,10,8,6,5,7,9,16,14,12,11,13,15,
]
coco_annotation_order_ext = coco_annotation_order+[
    17,18,19,20,21,22,
]
coco_oks_sigmas = np.asarray([
    0.26,  #  0, nose
    0.25,  #  1, eye_left
    0.25,  #  2, eye_right
    0.35,  #  3, ear_left
    0.35,  #  4, ear_right

    0.79,  #  5, shoulder_left
    0.79,  #  6, shoulder_right
    0.72,  #  7, elbow_left
    0.72,  #  8, elbow_right
    0.62,  #  9, wrist_left
    0.62,  # 10, wrist_right

    1.07,  # 11, hip_left
    1.07,  # 12, hip_right
    0.87,  # 13, knee_left
    0.87,  # 14, knee_right
    0.89,  # 15, ankle_left
    0.89,  # 16, ankle_right
])/10.0
coco_parts = [
    ( 4, 2),  #  0, ear_right - eye_right
    ( 2, 0),  #  1, eye_right - nose
    ( 0, 1),  #  2, nose - eye_left
    ( 1, 3),  #  3, eye_left - ear_left
    
    (10, 8),  #  4, wrist_right - elbow_right
    ( 8, 6),  #  5, elbow_right - shoulder_right
    ( 6, 5),  #  6, shoulder_right - shoulder_left
    ( 5, 7),  #  7, shoulder_left - elbow_left
    ( 7, 9),  #  8, elbow_left - wrist_left
    
    (16,14),  #  9, ankle_right - knee_right
    (14,12),  # 10, knee_right - hip_right
    (12,11),  # 11, hip_right - hip_left
    (11,13),  # 12, hip_left - knee_left
    (13,15),  # 13, knee_left- ankle_left
]
coco_parts_ext = coco_parts + [
    ( 0,17),  # 14, nose - nose_root
    ( 9,19),  # 15, wrist_left - thumb_left
    (10,20),  # 16, wrist_right - thumb_right
    (15,21),  # 17, ankle_left - toe_left
    (16,22),  # 18, ankle_right - toe_right
]
coco_part_colors = ucolors(len(coco_parts))
coco_part_colors_ext = coco_part_colors + ucolors(len(coco_parts_ext)-len(coco_parts))


######################## METRICS ########################

def pcp(gt_kps, gt_bbox, pred_kps, k=0.5, m=None, return_more=True):
    # convert to batched
    unsq = len(gt_kps.shape)==2
    if unsq:
        gt_kps = gt_kps[None]
        gt_bbox = gt_bbox[None]
        pred_kps = pred_kps[None]
    gt_kps = gt_kps[...,:2].float()
    gt_bbox = gt_bbox.float()
    pred_kps = pred_kps[...,:2]
    
    # normalize coordinates (wrt max bbox dim)
    bbox_from = gt_bbox[:,None,0]
    bbox_size = gt_bbox[:,1].max(dim=1).values[:,None,None]
    gt_kps = (gt_kps-bbox_from) / bbox_size
    pred_kps = (pred_kps-bbox_from) / bbox_size
    
    # threshold distance
    d = torch.norm(gt_kps-pred_kps, dim=2)
    if m is None:
        l = gt_kps[:,coco_parts]
        l = torch.norm(l[:,:,0]-l[:,:,1], dim=2)
    else:
        # m: 14-dim vector of mean lengths (norm wrt max bbox dim)
        l = m[None,:]
    yes = (d[:,coco_parts]<=k*l[...,None]).all(dim=-1)
    ans = {'metric': yes.float().mean(dim=1)}
    if return_more:
        ans['joint_distances'] = d
        ans['part_lengths'] = l
        ans['correct_parts'] = yes
        # ans['locals'] = locals()
    return ans
def pck(gt_kps, gt_bbox, pred_kps, k=0.5, mode='h', return_more=True):
    # convert to batched
    unsq = len(gt_kps.shape)==2
    if unsq:
        gt_kps = gt_kps[None]
        gt_bbox = gt_bbox[None]
        pred_kps = pred_kps[None]
    gt_kps = gt_kps[...,:2].float()
    gt_bbox = gt_bbox.float()
    pred_kps = pred_kps[...,:2]
    
    # normalize coordinates (wrt max bbox dim)
    bbox_from = gt_bbox[:,None,0]
    bbox_size = gt_bbox[:,1].max(dim=1).values[:,None,None]
    gt_kps = (gt_kps-bbox_from) / bbox_size
    pred_kps = (pred_kps-bbox_from) / bbox_size
    
    # threshold distance
    d = torch.norm(gt_kps-pred_kps, dim=2)
    if mode=='h':
        # pck(h)
        # approx. head size as clavicle to nose
        l = torch.norm(
            gt_kps[:,0]-(gt_kps[:,5]+gt_kps[:,6])/2,
            dim=1,
        )
    elif mode=='pdj':
        # pdj: hip_left - shoulder_right
        l = torch.norm(gt_kps[:,11]-gt_kps[:,6], dim=1)
    elif mode=='bbox_diag':
        # bbox diagonal
        l = torch.norm(gt_bbox[:,1]/bbox_size[:,0], dim=1)
    else:
        assert 0, f'mode {mode} not understood'
    yes = d<=k*l[:,None]
    ans = {'metric': yes.float().mean(dim=1)}
    if return_more:
        ans['joint_distances'] = d
        ans['joint_thresholds'] = l
        ans['correct_joints'] = yes
        # ans['locals'] = locals()
    return ans
def pdj(gt_kps, gt_bbox, pred_kps, k=0.2, return_more=True):
    return pck(gt_kps, gt_bbox, pred_kps, k=k, mode='pdj', return_more=return_more)
def oks(gt_kps, gt_bbox, pred_kps, thresh=0.5, return_more=True):
    # convert to batched
    unsq = len(gt_kps.shape)==2
    if unsq:
        gt_kps = gt_kps[None]
        gt_bbox = gt_bbox[None]
        pred_kps = pred_kps[None]
    gt_kps = gt_kps[...,:2].float()
    gt_bbox = gt_bbox.float()
    pred_kps = pred_kps[...,:2]
    
    # normalize coordinates (wrt max bbox dim)
    bbox_from = gt_bbox[:,None,0]
    bbox_size = gt_bbox[:,1].max(dim=1).values[:,None,None]
    gt_kps = (gt_kps-bbox_from) / bbox_size
    pred_kps = (pred_kps-bbox_from) / bbox_size
    # bbox_diag = torch.norm(gt_bbox[:,1]/bbox_size[:,0], dim=1)
    
    # threshold distance
    d = torch.norm(gt_kps-pred_kps, dim=2)
    # s = bbox_diag[:,None]
    s = torch.sqrt(gt_bbox[:,1,0]*gt_bbox[:,1,1] / bbox_size[:,0,0]**2)[:,None]
    sig = torch.tensor(coco_oks_sigmas, device=d.device)[None]
    ok = torch.exp(-d**2/(2*s**2*sig**2))
    yes = ok>=thresh
    ans = {'metric': yes.float().mean(dim=1)}
    if return_more:
        ans['joint_distances'] = d
        ans['keypoint_similarities'] = ok
        ans['correct_joints'] = yes
        ans['bbox_sqrt_area'] = s[:,0]
        # ans['locals'] = locals()
    return ans


######################## ANNOTATED DATASETS ########################

class DatabackendADDSKeypoints:
    def __init__(self, bargs, pargs, force_preprocess_mean_part_lengths=False):
        self.bargs = bargs
        self.pargs = pargs
        self.dn = f'{bargs.dn}/data/anime_drawings_dataset'
        self.fn_annotations = f'{self.dn}/raw/annotations.json'
        self.annotations = jread(self.fn_annotations)
        self.keyset = self.annotations['categories'][0]['name']

        # setup mappings
        cat_old = self.annotations['categories'][0]
        key_new2old = [
            cat_old['keypoints'].index(i)
            for i in coco_keypoints
        ]
        id2bn = {i['id']: int(i['file_name'].split('.')[0]) for i in self.annotations['images']}
        id2size = {i['id']: (i['height'],i['width']) for i in self.annotations['images']}

        # parse annotations
        self.bns = sorted(list(id2bn.values()))
        self.sizes = {
            id2bn[an['id']]: id2size[an['id']]
            for an in self.annotations['images']
        }
        self.keypoints = {
            id2bn[an['image_id']]: np.asarray([
                (k[1],k[0],k[2]) for k in chunk(an['keypoints'], 3)
            ])[key_new2old]
            for an in self.annotations['annotations']
            # if an['category_id']==1
            # and an['dataset_id']==1
        }
        self.bboxes = {
            id2bn[an['image_id']]: [
                (int(np.round(an['bbox'][1])), int(np.round(an['bbox'][0]))),
                (int(np.round(an['bbox'][3]-an['bbox'][1])), int(np.round(an['bbox'][2]-an['bbox'][0]))),
            ]
            for an in self.annotations['annotations']
            # if an['category_id']==1
            # and an['dataset_id']==1
        }

        # get mean part length (for pcpm)
        self.fn_mean_part_lengths = mkdir(f'{self.dn}/preprocessed/mean_part_lengths.pkl')
        if not force_preprocess_mean_part_lengths and os.path.isfile(self.fn_mean_part_lengths):
            self.mean_part_lengths = load(self.fn_mean_part_lengths)
        else:
            bns = self.keys()
            kps = []
            bxs = []
            for bn in bns:
                x = self[bn]
                kps.append(x['keypoints'][:,:2])
                bxs.append(np.asarray(x['bbox']))
            kps = torch.tensor(np.stack(kps))
            bxs = torch.tensor(np.asarray(bxs))
            out = pcp(kps, bxs, torch.zeros_like(kps))
            self.mean_part_lengths = out['part_lengths'].mean(0).numpy()
            dump(self.mean_part_lengths, self.fn_mean_part_lengths)
        return
    def __getitem__(self, bn, return_more=False):
        return {
            'bn': bn,
            'keypoints': self.keypoints[bn],
            'bbox': self.bboxes[bn],
            'size': self.sizes[bn],
        }
    def __len__(self):
        return len(self.bns)
    def __iter__(self):
        class it:
            def __init__(self, x):
                self.x = x
                self.idx = 0
            def __next__(self):
                i = self.idx
                if i>=len(self):
                    raise StopIteration
                self.idx += 1
                return self.x[self.x.keys()[i]]
            def __len__(self):
                return len(self.x)
        return it(self)
    def keys(self):
        return self.bns


# used only for refactoring into single annotation json
class DatabackendDanbooruCOCOKeypoints:
    def __init__(self, bargs, pargs, force_preprocess_mean_part_lengths=False):
        self.bargs = bargs
        self.pargs = pargs
        self.dn = f'{bargs.dn}/data/anime_drawings_dataset/danbooru_coco'
        self.fn_annotations = f'{self.dn}/raw/annotations.json'
        self.annotations = jread(self.fn_annotations)
        self.keyset = self.annotations['categories'][0]['name']

        # setup mappings
        cat_old = self.annotations['categories'][0]
        key_new2old = [
            cat_old['keypoints'].index(i)
            for i in coco_keypoints
        ]
        id2bn = {i['id']: int(i['file_name'].split('.')[0]) for i in self.annotations['images']}
        id2size = {i['id']: (i['height'],i['width']) for i in self.annotations['images']}

        # parse annotations
        self.bns = sorted(list(id2bn.values()))
        self.sizes = {
            id2bn[an['id']]: id2size[an['id']]
            for an in self.annotations['images']
        }
        self.keypoints = {
            id2bn[an['image_id']]: np.asarray([
                (k[1],k[0],k[2]) for k in chunk(an['keypoints'], 3)
            ])[key_new2old]
            for an in self.annotations['annotations']
            # if an['category_id']==1
            # and an['dataset_id']==1
        }
        self.bboxes = {
            id2bn[an['image_id']]: [
                (int(np.round(an['bbox'][1])), int(np.round(an['bbox'][0]))),
                (int(np.round(an['bbox'][3]-an['bbox'][1])), int(np.round(an['bbox'][2]-an['bbox'][0]))),
            ]
            for an in self.annotations['annotations']
            # if an['category_id']==1
            # and an['dataset_id']==1
        }

        # get mean part length (for pcpm)
        self.fn_mean_part_lengths = mkdir(f'{self.dn}/preprocessed/mean_part_lengths.pkl')
        if not force_preprocess_mean_part_lengths and os.path.isfile(self.fn_mean_part_lengths):
            self.mean_part_lengths = load(self.fn_mean_part_lengths)
        else:
            bns = self.keys()
            kps = []
            bxs = []
            for bn in bns:
                x = self[bn]
                kps.append(x['keypoints'][:,:2])
                bxs.append(np.asarray(x['bbox']))
            kps = torch.tensor(np.stack(kps))
            bxs = torch.tensor(np.asarray(bxs))
            out = util_keypoints.pcp(kps, bxs, torch.zeros_like(kps))
            self.mean_part_lengths = out['part_lengths'].mean(0).numpy()
            dump(self.mean_part_lengths, self.fn_mean_part_lengths)
        return
    def __getitem__(self, bn, return_more=False):
        return {
            'bn': bn,
            'keypoints': self.keypoints[bn],
            'bbox': self.bboxes[bn],
            'size': self.sizes[bn],
        }
    def __len__(self):
        return len(self.bns)
    def __iter__(self):
        class it:
            def __init__(self, x):
                self.x = x
                self.idx = 0
            def __next__(self):
                i = self.idx
                if i>=len(self):
                    raise StopIteration
                self.idx += 1
                return self.x[self.x.keys()[i]]
            def __len__(self):
                return len(self.x)
        return it(self)
    def keys(self):
        return self.bns






