from functools import partial
from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F
from utils import *

class WeightStandardizedConv2d(nn.Conv2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # define epsilon
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        """
        t = torch.ones([1, 4, 16, 16])
        m = reduce(t, 'b c h w -> b 1 1 1')
        m.shape = [1, 1, 1, 1]
        """
        mean = reduce(weight, "b c h w -> b 1 1 1", 'mean')
        var = reduce(weight, "b c h w -> b 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()   # torch.rsqrt()在元素级别上开方，不会影响tensor形状

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: list[torch.Tensor] | None = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return x
        
class ResNetBlock(nn.Module):
    def __init__(
            self, 
            dim_in: int, 
            dim_out: int, 
            *, 
            dim_time_emb: int | None = None, 
            groups: int = 8
        ) -> None:
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_time_emb, dim_out * 2) if exists(dim_time_emb) else None
            )
        )

        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class Attention(nn.Module):
    def __init__(self, dim_in: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim_in, dim_hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim_hidden, dim_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
    
class LinearAttention(nn.Module):
    def __init__(self, dim_in: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim_in, dim_hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim_hidden, dim_in, 1),
            nn.GroupNorm(1, dim_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
class PreNorm(nn.Module):
    def __init__(self, dim_in: int, fn: any) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)
    
class Unet(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int | None = None,
        init_dim: int | None = None,
        dim_mults: tuple[int] | list[int] = (1, 2, 4, 8),
        channels: int = 3,
        self_condition: bool = False,
        resnet_block_groups: int = 4,
    ) -> None:
        super().__init__()

        # define channels
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim_in)
        self.init_conv = nn.Conv2d(
            input_channels,
            init_dim,
            1,
            padding=0
        )   # from 7, 3 to 1, 0
        dims = [init_dim, *map(lambda m: dim_in * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResNetBlock, groups=resnet_block_groups)  # partial会提前定义函数的参数,这里是提前定义了groups

        # time embedding
        dim_time = dim_in * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim_in),
            nn.Linear(dim_in, dim_time),
            nn.GELU(),
            nn.Linear(dim_time, dim_time),
        )

        # layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        num_resolutions = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(d_in, d_in, dim_time_emb=dim_time),
                        block_klass(d_in, d_in, dim_time_emb=dim_time),
                        Residual(PreNorm(d_in, LinearAttention(d_in))),
                        Downsample(d_in, d_out) if not is_last else nn.Conv2d(d_in, d_out, 3, padding=1)
                    ]
                )
            )
        
        dim_mid = dims[-1]
        self.mid_block1 = block_klass(dim_mid, dim_mid, dim_time_emb=dim_time)
        self.mid_attn = Residual(PreNorm(dim_mid, Attention(dim_mid)))
        self.mid_block2 = block_klass(dim_mid, dim_mid, dim_time_emb=dim_time)

        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(d_out + d_in, d_out, dim_time_emb=dim_time),
                        block_klass(d_out + d_in, d_out, dim_time_emb=dim_time),
                        Residual(PreNorm(d_out, LinearAttention(d_out))),
                        Upsample(d_out, d_in) if not is_last else nn.Conv2d(d_out, d_in, 3, padding=1),
                    ]
                )
            )
        
        self.dim_out = default(dim_out, channels)
        
        self.final_res_block = block_klass(dim_in * 2, dim_in, dim_time_emb=dim_time)
        self.final_conv = nn.Conv2d(dim_in, self.dim_out, 1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, x_self_cond: torch.Tensor | None = None) -> torch.Tensor:
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

