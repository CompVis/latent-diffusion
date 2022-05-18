



from _util.util_v1 import *
from _util.pytorch_v1 import *


try:
    import igl
    import meshplot as mp # https://skoch9.github.io/meshplot/tutorial/
except:
    pass

try:
    import skimage
    from skimage import measure as _
    from skimage import color as _
    from skimage import segmentation as _
    from skimage import filters as _
    from scipy.spatial.transform import Rotation
except:
    pass

try:
    import colorsys
except:
    pass

try:
    import imagesize
except:
    pass

from PIL import Image, ImageFile, ImageFont, ImageDraw


################ NETWORK ################

def img2uri(img):
    bio = io.BytesIO()
    img.save(bio, 'PNG')
    return base64.b64encode(bio.getvalue())

def uri2img(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri)))


################ IMAGE HELPERS ################

# image wrapper
class I:
    # canonize
    def __init__(self, data):
        # preprocess stream-type to pil
        if isinstance(data, str):
            data = Image.open(data)
        elif isinstance(data, bytes):
            data = uri2img(data)
        self.data = data
        
        # massage to canonical forms
        if isinstance(self.data, Image.Image):
            # canon: pil image
            self.dtype = 'pil'
            self.mode = self.data.mode
            self.shape = (
                len(self.data.getbands()),
                self.data.size[1],
                self.data.size[0],
            )
            self.size = self.shape[1:]
        elif isinstance(self.data, np.ndarray):
            # canon: float(ch,h,w)
            if len(self.data.shape)==2:
                self.data = self.data[None,]
            elif len(self.data.shape)==4:
                assert self.data.shape[0]==1
                self.data = self.data[0]
            if self.data.shape[0] not in [1,3,4]:
                self.data = self.data.transpose(2,0,1)
            if np.issubdtype(self.data.dtype, np.floating):
                pass
            elif self.data.dtype==np.bool:
                self.data = self.data.astype(np.float)
            elif np.issubdtype(self.data.dtype, np.integer):
                self.data = self.data.astype(np.float) / 255.0
            self.dtype = 'np'
            self.mode = {
                1: 'L',
                3: 'RGB',
                4: 'RGBA',
            }[self.data.shape[0]]
            self.shape = self.data.shape
            self.size = self.shape[1:]
        elif isinstance(self.data, torch.Tensor):
            # canon: (ch,h,w)
            # assumes values in [0,1]
            if len(self.data.shape)==2:
                self.data = self.data[None,]
            elif len(self.data.shape)==4:
                assert self.data.shape[0]==1
                self.data = self.data[0]
            if self.data.shape[0] not in [1,3,4]:
                self.data = self.data.permute(2,0,1)
            self.dtype = 'torch'
            self.mode = {
                1: 'L',
                3: 'RGB',
                4: 'RGBA',
            }[self.data.shape[0]]
            self.shape = tuple(self.data.shape)
            self.size = self.shape[1:]
        elif isinstance(self.data, I):
            self.dtype = self.data.dtype
            self.mode = self.data.mode
            self.shape = self.data.shape
            self.size = self.data.size
            self.data = self.data.data
        else:
            assert 0, 'data not understood'
        self.diam = diam(self.size)
        return
    
    # conversion
    def convert(self, mode):
        return I(self.pil(mode=mode))
    def invert(self, invert_alpha=False):
        data = self.np()
        if self.mode=='RGBA' and not invert_alpha:
            return I(np.concatenate([
                1-data[:3], data[3:],
            ]))
        else:
            return I(1-self.np())
    def pil(self, mode=None):
        if self.dtype=='pil':
            ans = self.data
        elif self.dtype=='np':
            data = 255*self.data.clip(0,1)
            ans = Image.fromarray((
                data.transpose(1,2,0) if data.shape[0]!=1
                else data[0]
            ).astype(np.uint8))
        elif self.dtype=='torch':
            ans = TF.to_pil_image(self.data.float().clamp(0,1).cpu())
        else:
            assert 0, 'data not understood'
        return ans if mode is None else ans.convert(mode)
    def p(self, *args, **kwargs):
        return self.pil(*args, **kwargs)
    def pimg(self, mode=None):
        return self.pil(mode=mode)
    def np(self):
        if self.dtype=='pil':
            return I(np.asarray(self.data)).data
        elif self.dtype=='np':
            return self.data
        elif self.dtype=='torch':
            return self.data.cpu().numpy()
        assert 0, 'data not understood'
    def n(self):
        return self.np()
    def numpy(self):
        return self.np()
    def nimg(self):
        return self.np()
    def uint8(self, ch_last=True):
        ans = (self.np()*255).astype(np.uint8)
        return ans.transpose(1,2,0) if ch_last else ans
    def cv2(self):
        return self.uint8(ch_last=True)[...,::-1]
    def bgr(self):
        x = self.np()
        if self.mode=='RGBA':
            return I(x[[2,1,0,3]])
        else:
            return I(x[::-1])
    def tensor(self):
        if self.dtype=='pil':
            return TF.to_tensor(self.data)
        elif self.dtype=='np':
            return torch.from_numpy(self.data.copy())
        elif self.dtype=='torch':
            return self.data
        assert 0, 'data not understood'
    def t(self):
        return self.tensor()
    def torch(self):
        return self.tensor()
    def timg(self):
        return self.tensor()
    def save(self, fn):
        self.pil().save(fn)
        return fn
    
    # resizing
    def rescale(self, factor, resample='bilinear', antialias=False):
        return self.resize(
            rescale_dry(self.size, factor),
            resample=resample, antialias=antialias,
        )
    def resize(self, s, resample='bilinear', antialias=False):
        s = pixel_ij(s, rounding=True)
        if self.dtype=='pil':
            return I(self.data.resize(
                s[::-1], resample=getattr(Image, resample.upper()),
            ))
        elif self.dtype=='np':
            return I(self.pil()).resize(s, resample=resample)
        elif self.dtype=='torch':
            return I(TF.resize(
                self.data,
                s,
                interpolation=getattr(TF.InterpolationMode, resample.upper()),
                # antialias=antialias,
            ))
        assert 0, 'data not understood'
    def resize_w(self, s=512, resample='bilinear', antialias=False):
        h,w = self.size
        return self.resize((h*s/w, s))
    def resize_h(self, s=512, resample='bilinear', antialias=False):
        h,w = self.size
        return self.resize((s, w*s/h))
    def resize_max(self, s=512, resample='bilinear', antialias=False):
        dry = resize_max_dry(self.size, s=s)
        return self.resize(dry, resample=resample)
    def resize_min(self, s=512, resample='bilinear', antialias=False):
        dry = resize_min_dry(self.size, s=s)
        return self.resize(dry, resample=resample)
    def resize_square(
                self, s=512,
                resample='bilinear', antialias=False,
                fill=0, padding_mode='constant',
            ):
        res = self.resize_max(s=s, resample=resample, antialias=antialias).tensor()
        dry = resize_square_dry(res, s=s)
        fc = dry[0]
        pad = TF.pad(
            res,
            padding=[
                -fc[1], # left
                -fc[0], # top
                s+fc[1]-res.shape[2], # right
                s+fc[0]-res.shape[1], # bottom
            ],
            fill=fill, padding_mode=padding_mode,
        )
        return I(pad)

    def rw(self, *args, **kwargs):
        return self.resize_w(*args, **kwargs)
    def rh(self, *args, **kwargs):
        return self.resize_h(*args, **kwargs)
    def rmax(self, s=512, resample='bilinear', antialias=False):
        return self.resize_max(s=s, resample=resample, antialias=antialias)
    def rmin(self, s=512, resample='bilinear', antialias=False):
        return self.resize_min(s=s, resample=resample, antialias=antialias)
    def rsqr(self, *args, **kwargs):
        return self.resize_square(*args, **kwargs)

    # transformation
    def transpose(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.TRANSPOSE))
        elif self.dtype=='np':
            return I(np.swapaxes(self.data, 1, 2))
        elif self.dtype=='torch':
            return I(self.data.permute(0,2,1))
        assert 0, 'data not understood'
    def T(self):
        return self.transpose()
    def fliph(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.FLIP_LEFT_RIGHT))
        elif self.dtype=='np':
            return I(self.data[...,::-1])
        elif self.dtype=='torch':
            return I(self.data.flip(dims=(2,)))
        assert 0, 'data not understood'
    def flipv(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.FLIP_TOP_BOTTOM))
        elif self.dtype=='np':
            return I(self.data[:,::-1])
        elif self.dtype=='torch':
            return I(self.data.flip(dims=(1,)))
        assert 0, 'data not understood'
    def rotate(self, deg):
        if deg==0:
            return self
        elif deg==90:
            return self.rotate90()
        elif deg==180:
            return self.rotate180()
        elif deg==270:
            return self.rotate270()
        elif deg==360:
            return self
        assert 0, 'data not understood'
    def rotate90(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.ROTATE_90))
        elif self.dtype in ['np', 'torch']:
            return self.transpose().flipv()
    def rotate180(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.ROTATE_180))
        elif self.dtype in ['np', 'torch']:
            return self.fliph().flipv()
    def rotate270(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.ROTATE_270))
        elif self.dtype in ['np', 'torch']:
            return self.transpose().fliph()

    # cropping
    def cropbox(self, from_corner, from_size, to_size=None, resample='bilinear'):
        from_corner = pixel_ij(from_corner, rounding=True)
        from_size = pixel_ij(from_size, rounding=True)
        to_size = pixel_ij(to_size, rounding=True) if to_size!=None else from_size
        return I(TF.resized_crop(
            self.pil().convert('RGBA'),
            from_corner[0],
            from_corner[1],
            from_size[0],
            from_size[1],
            to_size,
            interpolation=getattr(TF.InterpolationMode, resample.upper()),
        ))
    def cb(self, *args, **kwargs):
        return self.cropbox(*args, **kwargs)

    # transparency
    def alpha_composite(self, img, opacity=1.0):
        a = self.pil().convert('RGBA')
        b = I(img).pil().convert('RGBA')
        if opacity==0:
            return I(a)
        # elif opacity==1:
        #     return I(b)
        else:
            b = I(b).np() * np.asarray([1,1,1,opacity])[:,None,None]
            b = I(b).pil()
            return I(Image.alpha_composite(a,b))
    def alpha_bg(self, c=0.5):
        return iblank(self.size, c=c).alpha_composite(self, opacity=1.0)
    def alpha_bbox(self, thresh=0.5):
        return alpha_bbox(self, thresh=thresh)
    def as_alpha(self, c=0):
        return iblank(self.size, c=c).alpha(self)
    def alpha(self, a=1.0):
        # setter, converts to rgba
        rgba = I(self.pil().convert('RGBA'))
        a = I(a*np.ones(self.size)) if type(a) in [float, int] else I(a)
        return I(np.concatenate([
            rgba.numpy()[:-1], a.numpy()[-1:],
        ]))

    def acomp(self, *args, **kwargs):
        return self.alpha_composite(*args, **kwargs)
    def abg(self, *args, **kwargs):
        return self.alpha_bg(*args, **kwargs)
    def abbox(self, *args, **kwargs):
        return self.alpha_bbox(*args, **kwargs)
    def aa(self, *args, **kwargs):
        return self.as_alpha(*args, **kwargs)
    
    # compositing
    def left(self, img, bg='k'):
        return igrid([img, self], bg=bg)
    def right(self, img, bg='k'):
        return igrid([self, img], bg=bg)
    def top(self, img, bg='k'):
        return igrid([[img,], [self,]], bg=bg)
    def bottom(self, img, bg='k'):
        return igrid([[self,], [img,]], bg=bg)

    # drawing
    def rect(self, corner, size, w=1, c='r', f=None):
        corner = pixel_ij(corner, rounding=True)
        size = pixel_ij(size, rounding=True)
        w = max(1, round(w))
        c = c255(c)
        f = c255(f)
        ans = self.pil(mode='RGBA').copy()
        d = ImageDraw.Draw(ans)
        d.rectangle(
            [corner[1], corner[0], corner[1]+size[1]-1, corner[0]+size[0]-1],
            fill=f, outline=c, width=w,
        )
        return I(ans)
    def bbox(self, *args, **kwargs):
        return self.rect(*args, **kwargs)
    def border(self, w=1, c='r'):
        return self.rect(
            (0, 0),
            self.size,
            w=w, c=c, f=None,
        )
    def dot(self, point, s=1, c='r'):
        c = c255(c)
        x,y = pixel_ij(point, rounding=False)
        ans = self.pil(mode='RGBA').copy()
        d = ImageDraw.Draw(ans)
        d.ellipse(
            [(y-s,x-s), (y+s,x+s)],
            fill=c,
        )
        return I(ans)
    def point(self, *args, **kwargs):
        return self.dot(*args, **kwargs)
    def line(self, a, b, w=1, c='r'):
        a = pixel_ij(a, rounding=False)
        b = pixel_ij(b, rounding=False)
        c = c255(c)
        w = max(1, round(w))
        ans = self.pil(mode='RGBA').copy()
        d = ImageDraw.Draw(ans)
        d.line([a[::-1], (b[1]-1,b[0]-1)], fill=c, width=w)
        return I(ans)

    # text
    def text(self, text, pos, s=12, anchor='tl', c='m', bg='k', spacing=None, padding=0):
        t = itext(
            text, s=s, c=c, bg=bg,
            spacing=spacing, padding=padding,
        )
        x,y = pos
        x = {
            't': x,
            'b': x-t.size[0],
            'c': x-t.size[0]/2,
        }[anchor[0].lower()]
        y = {
            'l': y,
            'r': y-t.size[1],
            'c': y-t.size[1]/2,
        }[anchor[1].lower()]
        t = t.pil('RGBA')
        ans = self.pil('RGBA')
        ans.paste(t, pixel_ij((y,x), rounding=True), t)
        return I(ans)
    def caption(self, text, s=24, pos='t', c='w', bg='k', spacing=None, padding=None):
        pos = pos[0].lower()
        t = itext(text, s=s, c=c, bg=bg, spacing=spacing, padding=padding)
        if pos=='t':
            return self.top(t)
        elif pos=='b':
            return self.bottom(t)
        elif pos=='l':
            return self.left(t)
        elif pos=='r':
            return self.right(t)
        assert 0, 'data not understood'
    def cap(self, *args, **kwargs):
        return self.caption(*args, **kwargs)

    # ipython integration
    def _repr_png_(self):
        bio = io.BytesIO()
        self.pil().save(bio, 'PNG')
        return bio.getvalue()


# conversion
def pimg(x):
    return I(x).pil()
def nimg(x):
    return I(x).numpy()
def timg(x):
    return I(x).tensor()

# resizing
def pixel_rounder(n, mode):
    if mode==True or mode=='round':
        return round(n)
    elif mode=='ceil':
        return math.ceil(n)
    elif mode=='floor':
        return math.floor(n)
    else:
        return n
def pixel_ij(x, rounding=True):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return tuple(pixel_rounder(i, rounding) for i in (
        x if isinstance(x, tuple) or isinstance(x, list) else (x,x)
    ))
def diam(x):
    if isinstance(x, tuple) or isinstance(x, list):
        h,w = x[-2:]
    elif isinstance(x, I):
        h,w = x.size
    else:
        h,w = x.shape[-2:]
    return np.sqrt(h**2 + w**2)
def rescale(x, factor, resample='bilinear', antialias=False):
    return x.rescale(factor, resample=resample, antialias=antialias)
def rescale_dry(x, factor):
    h,w = x[-2:] if isinstance(x, tuple) or isinstance(x, list) else I(x).size
    return (h*factor, w*factor)
def resize_max(x, s=512, resample='bilinear', antialias=False):
    return I(x).resize_max(s=s, resample=resample, antialias=antialias)
def resize_max_dry(x, s=512):
    # returns size
    h,w = x[-2:] if isinstance(x, tuple) or isinstance(x, list) else I(x).size
    return (
        (s, int(w*s/h)),
        (int(h*s/w), s),
    )[h<w]
def resize_min(x, s=512, resample='bilinear', antialias=False):
    return I(x).resize_min(s=s, resample=resample, antialias=antialias)
def resize_min_dry(x, s=512):
    # returns size
    h,w = x[-2:] if isinstance(x, tuple) or isinstance(x, list) else I(x).size
    return (
        (s, int(w*s/h)),
        (int(h*s/w), s),
    )[w<h]
def resize_square(
            x, s=512,
            resample='bilinear', antialias=False,
            fill=0, padding_mode='constant',
        ):
    return I(x).resize_square(
        s=s, resample=resample, antialias=antialias,
        fill=fill, padding_mode=padding_mode,
    )
def resize_square_dry(x, s=512):
    # returns a forward cropbox
    h,w = x[-2:] if isinstance(x, tuple) or isinstance(x, list) else I(x).size
    from_corner = (
        (0, -(h-w)//2),
        (-(w-h)//2, 0),
    )[h<w]
    from_size = (max(h,w),)*2
    to_size = (s, s)
    return (from_corner, from_size, to_size)

# cropping
def cropbox(x, from_corner, from_size, to_size=None, resample='bilinear'):
    return I(x).cropbox(
        from_corner, from_size, to_size, resample=resample,
    )
def cropbox_compose(cba, cbb):
    # compose two cropboxes
    fca,fsa,tsa = [pixel_ij(q, rounding=False) for q in cba]
    fcb,fsb,tsb = [pixel_ij(q, rounding=False) for q in cbb]
    sfx = fsa[0] / tsa[0]
    sfy = fsa[1] / tsa[1]
    fc = fca[0]+fcb[0]*sfx, fca[1]+fcb[1]*sfy
    fs = fsb[0]*sfx, fsb[1]*sfy
    ts = tsb
    return fc, fs, ts
def cropbox_sequence(cropboxes):
    # compose multiple cropboxes in sequence
    ans = cropboxes[-1]
    for c in range(len(cropboxes)-2, -1, -1):
        cb = cropboxes[c]
        ans = cropbox_compose(cb, ans)
    return ans
def cropbox_points(pts, from_corner, from_size, to_size):
    # apply cropbox to points
    pts = np.asarray(pts)
    assert len(pts.shape)==2 and pts.shape[1]==2
    fc = pixel_ij(from_corner, rounding=False)
    fs = pixel_ij(from_size, rounding=False)
    ts = pixel_ij(to_size, rounding=False)
    fc = np.asarray(fc)[None,]
    sf = np.asarray([ts[0]/fs[0], ts[1]/fs[1]])[None,]
    return (pts-fc)*sf
def cropbox_bbox(bbox, from_corner, from_size, to_size):
    # apply cropbox to bbox
    pts = [
        bbox[0],
        (bbox[1][0]+bbox[0][0], bbox[1][1]+bbox[0][1]),
    ]
    ans = cropbox_points(pts, from_corner, from_size, to_size)
    return [
        (float(ans[0,0]), float(ans[0,1])),
        (float(ans[1,0]-ans[0,0]), float(ans[1,1]-ans[0,1])),
    ]
def cropbox_inverse(origin_size, from_corner, from_size, to_size):
    # origin_size: original image size
    # from_corner/from_size/to_size: of cropbox to invert
    origin_size = pixel_ij(origin_size, rounding=False)
    from_corner = pixel_ij(from_corner, rounding=False)
    from_size = pixel_ij(from_size, rounding=False)
    to_size = pixel_ij(to_size, rounding=False)
    sx,sy = to_size[0]/from_size[0], to_size[1]/from_size[1]
    return [
        (-from_corner[0]*sx, -from_corner[1]*sy),
        (origin_size[0]*sx, origin_size[1]*sy),
        origin_size,
    ]
def cropbox_bbox_square(bbox, s=512, padding=0):
    # focuses + zooms out
    # padding is additional to s
    pd = padding
    fins = s + 2*padding
    return cropbox_sequence([
        [bbox[0], bbox[1], bbox[1]],
        resize_square_dry(bbox[1], s=s),
        [(-pd,-pd), (fins,fins), (fins,fins)],
    ])
def cropbox_borders(from_size, top_bottom, left_right):
    # returns cropbox that removes borders
    # from_size: size of image to crop
    h,w = pixel_ij(from_size, rounding=False)
    t,b = pixel_ij(top_bottom, rounding=False)
    l,r = pixel_ij(left_right, rounding=False)
    return [
        (t, l),  # from_corner
        (h-t-b, w-l-r), # from_size
        (h-t-b, w-l-r), # to_size
    ]
def cropbox_resize(from_size, to_size):
    # dummy forward cropbox of resize operation
    return [
        (0, 0),  # from_corner
        pixel_ij(from_size, rounding=False),  # from_size
        pixel_ij(to_size, rounding=False),  # to_size
    ]
def cropbox_to_mask(origin_size, from_corner, from_size, conservative=True):
    s = pixel_ij(origin_size, rounding=True)
    if conservative:
        # smaller active area, do rounding myself
        x,y = pixel_ij(from_corner, rounding=False)
        h,w = pixel_ij(from_size, rounding=False)
        t,b = max(0, math.ceil(x)), min(math.floor(x+h), s[0])
        l,r = max(0, math.ceil(y)), min(math.floor(y+w), s[1])
    else:
        x,y = pixel_ij(from_corner, rounding=True)
        h,w = pixel_ij(from_size, rounding=True)
        t,b = max(0, x), min(x+h, s[0])
        l,r = max(0, y), min(y+w, s[1])
    ans = np.zeros(s)
    ans[t:b,l:r] = 1
    return ans[None,]
def bbox_lim(bbox, xlim=None, ylim=None, blim=None):
    # box limits
    if blim is not None:
        assert xlim is None and ylim is None
        (x,y),(h,w) = blim
        return bbox_lim(bbox, xlim=(x,x+h), ylim=(y,y+w))
        
    # x or y-range limits
    else:
        assert xlim is not None or ylim is not None
        (a,b),(h,w) = bbox
        u,v = a+h, b+w
        if xlim is not None:
            if isinstance(xlim, tuple) or isinstance(xlim, list):
                x0,x1 = xlim
            else:
                x0 = x1 = xlim
            a = np.clip(a, a_min=x0, a_max=x1)
            u = np.clip(u, a_min=x0, a_max=x1)
        if ylim is not None:
            if isinstance(ylim, tuple) or isinstance(ylim, list):
                y0,y1 = ylim
            else:
                y0 = y1 = ylim
            b = np.clip(b, a_min=y0, a_max=y1)
            v = np.clip(v, a_min=y0, a_max=y1)
        return (a,b),(u-a,v-b)

# color
def c255(c):
    # color format utility
    if c is None:
        return None
    if isinstance(c, str):
        c = {
            'r': (1,0,0),
            'g': (0,1,0),
            'b': (0,0,1),
            'k': 0,
            'w': 1,
            't': (0,1,1),
            'm': (1,0,1),
            'y': (1,1,0),
            'a': (0,0,0,0),
        }[c]
    if isinstance(c, list) or isinstance(c, tuple):
        if len(c)==3:
            c = c + (1,)
        elif len(c)==1:
            c = (c,c,c,1)
        c = tuple(int(255*q) for q in c)
    else:
        c = int(255*c)
        c = (c,c,c,255)
    return c
def ucolors(num_colors):
    # uniform color generator
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

# compositing
def iblank(size, c=(0,0,0,1)):
    size = pixel_ij(size, rounding=True)
    assert max(size)<4098*2
    if c is None: c = 'a'
    c = c255(c)
    return I(Image.fromarray(
        np.asarray(c, dtype=np.uint8)[None,None]
    )).resize(size, resample='nearest')
def alpha_composite(a, b, opacity=1.0):
    return I(a).alpha_composite(b, opacity=opacity)
def alpha_bg(x, c=0.5):
    return I(x).alpha_bg(c=c)
def alpha_bbox(img, thresh=0.5):
    h,w = img.size
    rgba = img.np()
    assert len(rgba) in [1,4]
    a = rgba[-1]
    x = np.max(a, axis=1)>thresh
    y = np.max(a, axis=0)>thresh
    whx = np.where(x)[0]
    why = np.where(y)[0]
    x0,x1 = (whx.min(),whx.max()+1) if len(whx)>0 else (0,h)
    y0,y1 = (why.min(),why.max()+1) if len(why)>0 else (0,w)
    fc = x0, y0
    s = x1-x0, y1-y0
    return fc, s
def igrid(imgs, just=True, bg='k'):
    # repackage
    assert isinstance(imgs, list)
    if any(isinstance(i, list) for i in imgs):
        x = [
            [j for j in i] if isinstance(i, list) else [i,]
            for i in imgs
        ]
    else:
        x = [[i for i in imgs],]

    # get sizes
    nrows = len(x)
    ncols = max(len(row) for row in x)
    hs = np.zeros((nrows,ncols))
    ws = np.zeros((nrows,ncols))
    for r in range(nrows):
        row = x[r]
        for c in range(ncols):
            if c==len(row): row.append(None)
            item = row[c]
            if item is None:
                hs[r,c] = ws[r,c] = 0
            else:
                item = I(item)
                hs[r,c], ws[r,c] = item.size
            row[c] = item
    offx = np.cumsum(np.max(hs, axis=1))
    if just:
        offy = np.cumsum(np.max(ws, axis=0))[None,...].repeat(nrows,0)
    else:
        offy = np.cumsum(ws, axis=1)

    # composite
    ans = Image.new('RGBA', (int(offy.max()), int(offx[-1])))
    for r in range(nrows):
        for c in range(ncols):
            item = x[r][c]
            if item is not None:
                ox = offx[r-1] if r>0 else 0
                oy = offy[r,c-1] if c>0 else 0
                ans.paste(item.pil(mode='RGBA'), (int(oy),int(ox)))
    ans = I(ans).alpha_bg(bg)
    return ans

# text
FN_ARIAL = './env/arial_monospaced_mt.ttf'
if not os.path.isfile(FN_ARIAL):
    FN_ARIAL = './__env__/arial_monospaced_mt.ttf'
if not os.path.isfile(FN_ARIAL):
    FN_ARIAL = './_env/arial_monospaced_mt.ttf'
def itext(
            text,
            s=24,
            facing='right',  # write in this direction
            pos='left',      # align to this position
            c='w',
            bg='k',
            h=None,
            w=None,
            spacing=None,  # between lines
            padding=None,  # around entire thing
            force_size=False,
        ):
    # text image utility
    text = str(text)
    s = max(1, round(s))
    spacing = math.ceil(s*4/10) if spacing is None else spacing
    padding = math.ceil(s*4/10) if padding is None else padding
    facing = facing.lower()
    if facing in ['u', 'up', 'd', 'down']:
        h,w = w,h
    c,bg = c255(c), c255(bg)
    f = PIL.ImageFont.truetype(FN_ARIAL, s)

    td = PIL.ImageDraw.Draw(Image.new('RGBA', (1,1), (0,0,0,0)))
    tw,th = td.multiline_textsize(text, font=f, spacing=spacing)
    if not force_size:
        if h and h<th: h = th
        if w and w<tw: w = tw
    h = h or th+2*padding
    w = w or tw+2*padding

    pos = pos.lower()
    an = None
    if pos in ['c', 'center']:
        xy = (w//2, h//2)
        an = 'mm'
        align = 'center'
    elif pos in ['l', 'lc', 'cl', 'left']:
        xy = (padding, h//2)
        an = 'lm'
        align = 'left'
    elif pos in ['r', 'rc', 'cr', 'right']:
        xy = (w-padding, h//2)
        an = 'rm'
        align = 'right'
    elif pos in ['t', 'tc', 'ct', 'top']:
        xy = (w//2, padding)
        an = 'ma'
        align = 'center'
    elif pos in ['b', 'bc', 'cb', 'bottom']:
        xy = (w//2, h-padding)
        an = 'md'
        align = 'center'
    elif pos in ['tl', 'lt']:
        xy = (padding, padding)
        align = 'left'
    elif pos in ['bl', 'lb']:
        xy = (padding, h-padding-th)
        align = 'left'
    elif pos in ['tr', 'rt']:
        xy = (w-padding-tw, padding)
        align = 'right'
    elif pos in ['br', 'rb']:
        xy = (w-padding-tw, h-padding-th)
        align = 'right'
    else:
        assert False, 'pos not understood'
    
    ans = Image.new('RGBA', (w,h), bg)
    d = PIL.ImageDraw.Draw(ans)
    d.multiline_text(
        xy, text, fill=c, font=f, anchor=an,
        spacing=spacing, align=align,
    )

    if facing in ['l', 'left']:
        ans = ans.rotate(180)
    elif facing in ['u', 'up']:
        ans = ans.rotate(90, expand=True)
    elif facing in ['d', 'down']:
        ans = ans.rotate(-90, expand=True)
    return I(ans)

# pixel-safe logit
# intended as safe sigmoid inverse of [0,1] images
# margin=1 ==> (+/-)5.5373 max/min logit
def pixel_logit(x, pixel_margin=1):
    x = (x*(255-2*pixel_margin) + pixel_margin) / 255
    return torch.log(x / (1-x))



