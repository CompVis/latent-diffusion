import torch as th
import torch.nn as nn

from omegaconf import ListConfig
from typing import List, Union, Tuple

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import Downsample, Upsample, ResBlock
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential as TimeEmbSeq
from ldm.modules.diffusionmodules.util import linear, zero_module
from ldm.modules.diffusionmodules.util import conv_nd, normalization, timestep_embedding


class ContextUNetModel(nn.Module):
    '''
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output tensor.
    :param num_res_blocks: number of residual blocks for each level of the UNet.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
        a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param xf_ctx_dim: number of layers for spatial transformer, if none,
        use self-attn instead.
    :param sp_xf_ctx_dim: dimensions of context
    '''

    def __init__(
        self,
        in_channels:int,
        model_channels:int,
        out_channels:int,
        num_res_blocks:List[int],
        attention_resolutions:List[int],
        dropout:float=0,
        channel_mult:List[int]=[1,2,4,8],
        conv_resample:bool=True,
        dims:int=2,
        num_classes:int=None,
        use_checkpoint:bool=False,
        num_heads:int=-1,
        num_head_channels:int=-1,
        use_scale_shift_norm:bool=False,
        resblock_updown:bool=False,
        transformer_depth:int=None,
        context_dim:Union[int,ListConfig]=None,
    ) -> None:
        super().__init__()
        assert transformer_depth is not None
        assert context_dim is not None

        if type(context_dim) == ListConfig:
            context_dim = list(context_dim)

        assert num_heads != -1 or num_head_channels != -1
        num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.resblock_updown = resblock_updown
        self.use_scale_shift_norm = use_scale_shift_norm

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.i_block = TimeEmbSeq(conv_nd(dims, in_channels, model_channels, 3, padding=1))

        d_block_chans = []
        self.d_blocks = nn.ModuleList()
        in_ch = model_channels
        down_scl, lvl = 1, 0
        for mult in channel_mult:
            out_ch = mult*model_channels
            down_scl *= 2
            blocks, chans = self.make_dblock(
                in_ch=in_ch,
                out_ch=out_ch,
                need_down=True,
                down_scl=down_scl,
                time_embed_dim=time_embed_dim,
                num_res_blocks=num_res_blocks[lvl])
            self.d_blocks.extend(blocks)
            d_block_chans.extend(chans)
            in_ch = out_ch
            lvl += 1

        self.e_block = TimeEmbSeq(
            ResBlock(
                out_ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                out_ch,
                num_heads,
                num_head_channels if num_head_channels != -1 else out_ch // num_heads,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResBlock(
                out_ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.u_blocks = nn.ModuleList([])
        channel_mult = [1] + channel_mult[:-1]
        for mult in channel_mult[::-1]:
            lvl -= 1
            out_ch = mult*model_channels
            blocks = self.make_ublock(
                in_ch=in_ch,
                out_ch=out_ch,
                need_up=True,
                down_scl=down_scl,
                time_embed_dim=time_embed_dim,
                input_block_chans=d_block_chans,
                num_res_blocks=num_res_blocks[lvl])
            self.u_blocks.extend(blocks)
            in_ch = out_ch
            down_scl //= 2

        self.out = nn.Sequential(
            normalization(out_ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def make_ublock(
        self,
        in_ch:int,
        out_ch:int,
        need_up:bool,
        down_scl:int,
        time_embed_dim:int,
        input_block_chans:List[int],
        num_res_blocks:int,
    ) -> nn.ModuleList:
        dims = self.dims
        dropout = self.dropout
        context_dim = self.context_dim
        conv_resample = self.conv_resample
        use_checkpoint = self.use_checkpoint
        resblock_updown = self.resblock_updown
        transformer_depth = self.transformer_depth
        use_scale_shift_norm = self.use_scale_shift_norm
        attention_resolutions = self.attention_resolutions
        num_head_channels = self.num_head_channels

        blocks = nn.ModuleList()
        for i in range(num_res_blocks + 1):
            skip_ch = input_block_chans.pop()
            layers = [ResBlock(
                in_ch+skip_ch,
                time_embed_dim,
                dropout,
                out_channels=in_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )]
            if down_scl in attention_resolutions:
                if num_head_channels == -1:
                    num_heads = self.num_heads
                    dim_head = in_ch // num_heads
                else:
                    num_heads = in_ch // num_head_channels
                    dim_head = num_head_channels
                layers.append(SpatialTransformer(
                    in_ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                ))
            if need_up and i == num_res_blocks:
                layers.append(ResBlock(
                    in_ch,
                    time_embed_dim,
                    dropout,
                    out_channels=out_ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    up=True,
                )
                if resblock_updown else Upsample(
                    in_ch,
                    conv_resample,
                    dims=dims,
                    out_channels=out_ch,
                ))
            blocks.append(TimeEmbSeq(*layers))
        return blocks


    def make_dblock(
        self,
        in_ch:int,
        out_ch:int,
        need_down:bool,
        down_scl:int,
        time_embed_dim:int,
        num_res_blocks:int
    ) -> Tuple[nn.ModuleList,List[int]]:
        dims = self.dims
        dropout = self.dropout
        context_dim = self.context_dim
        conv_resample = self.conv_resample
        use_checkpoint = self.use_checkpoint
        resblock_updown = self.resblock_updown
        transformer_depth = self.transformer_depth
        use_scale_shift_norm = self.use_scale_shift_norm
        attention_resolutions = self.attention_resolutions
        num_head_channels = self.num_head_channels

        blocks = nn.ModuleList()
        channs = []
        if need_down:
            blocks.append(TimeEmbSeq(
                ResBlock(
                    in_ch,
                    time_embed_dim,
                    dropout,
                    out_channels=out_ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=True,
                ) if resblock_updown else Downsample(
                    in_ch,
                    conv_resample,
                    dims=dims,
                    out_channels=out_ch,
                )
            ))
            channs.append(out_ch)

        layers = []
        for _ in range(num_res_blocks):
            layers.append(ResBlock(
                out_ch,
                time_embed_dim,
                dropout,
                out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ))
            if down_scl in attention_resolutions:
                if num_head_channels == -1:
                    num_heads = self.num_heads
                    dim_head = out_ch // num_heads
                else:
                    num_heads = out_ch // num_head_channels
                    dim_head = num_head_channels
                layers.append(SpatialTransformer(
                    out_ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                ))
            blocks.append(TimeEmbSeq(*layers))
            channs.append(out_ch)

        return blocks, channs

    def forward(
        self,
        x:th.Tensor,
        timesteps:th.Tensor=None,
        context:th.Tensor=None,
        **kwargs
    ):
        '''
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        '''
        hs = []
        model_channels = self.model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x
        h = self.i_block(h, emb, context)
        for module in self.d_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.e_block(h, emb, context)
        for module in self.u_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)


if __name__ == '__main__':
    device = th.device('cuda')
    model = ContextUNetModel(
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=[2,2,2,2],
        attention_resolutions=[8,4,2],
        dropout=0.2,
        use_checkpoint=True,
        channel_mult=[1,2,3,4],
        num_heads=8,
        transformer_depth=1,
        context_dim=1024,
    ).to(device)
    model.eval()
    inputs = th.rand((2, 3, 128, 128), device=device)
    ts = th.ones((2,), device=device)
    ctxs = th.rand((2, 512, 1024), device=device)
    outs = model(inputs, ts, ctxs)
    print(outs.shape)
