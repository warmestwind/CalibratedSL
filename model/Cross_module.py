import torch
from torch import nn, einsum
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv, y_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        kx = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        vx = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv)
        qx = rearrange(q, 'b n (h d) -> b h n d', h = h)


        k = self.to_k(y_qkv)
        ky = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(y_qkv)
        vy = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(y_qkv)
        qy = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', qx, ky) * self.scale
        attnx = dots.softmax(dim=-1)

        outx = einsum('b h i j, b h j d -> b h i d', attnx, vy)
        outx = rearrange(outx, 'b h n d -> b n (h d)')
        outx = self.to_out(outx)

        dots = einsum('b h i d, b h j d -> b h i j', qy, kx) * self.scale
        attny = dots.softmax(dim=-1)

        outy = einsum('b h i j, b h j d -> b h i d', attny, vx)
        outy = rearrange(outy, 'b h n d -> b n (h d)')
        outy = self.to_out(outy)

        outx = rearrange(outx, 'b (h w) d -> b d h w', h=int(n**0.5), w=int(n**0.5))
        outy = rearrange(outy, 'b (h w) d -> b d h w', h=int(n ** 0.5), w=int(n**0.5))

        return outx, outy
