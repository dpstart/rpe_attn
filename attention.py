import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class RPE_NKA(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        nb_features=None,
        kernel_fn=nn.ELU(),
        dropout=0.0,
        qkv_bias=False,
        attn_out_bias=True,
        max_len=1024,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        nb_features = default(nb_features, int(dim_head * math.log(dim_head)))

        self.max_len = max_len
        self.Er = nn.Parameter(torch.randn(max_len, dim_head))

        self.heads = heads
        self.dim_heads = dim_head
        self.nb_features = nb_features

        self.kernel_fn = kernel_fn

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x, context=None,
    ):

        b, n, _ = x.shape
        context = default(context, x)

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        q, k = map(lambda t: t / torch.norm(t), (q, k))
        k_, q_ = map(lambda t: self.kernel_fn(t), (k, q))

        start = self.max_len - n
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        b = self.skew(QEr)
        c = torch.exp(b)

        a_1 = torch.einsum("b h n x,b h n y->b h n x y", k_, v)
        a_1 = rearrange(a_1, "b h n x y -> b h n (x y)")

        a_2 = k_

        d_2 = fft_multiply_parallel(c, a_2)
        d_1 = fft_multiply_parallel(c, a_1)

        d_1 = rearrange(
            d_1, "b h n (x y) -> b h n x y", x=self.dim_heads, y=self.dim_heads
        )
        top = torch.einsum("b h n x, b h n y z -> b h n z", q_, d_1)
        bottom = torch.einsum("b h n x, b h n y -> b h n y", q_, d_2)

        out = top / bottom
        return out

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


def fft_multiply_parallel(A, x):

    b, h, n, d = x.shape

    # Select indices of first columns
    idx = [i for i in range(A.size(-1) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    inverted_tensor_1 = A.index_select(-1, idx)

    # Select indices of first columns + x_0 at beginning
    idx = [0] + [i for i in range(A.size(-1) - 1, 0, -1)]
    idx = torch.LongTensor(idx)
    inverted_tensor_2 = A.index_select(-2, idx)
    y = torch.cat(
        [inverted_tensor_1[:, :, :, 0], inverted_tensor_2[:, :, 0, :]], dim=-1
    )

    zeros = torch.zeros(*x.shape[:-2], y.size(-1) - x.size(-2), x.shape[-1]).float()
    v = torch.cat((x, zeros), dim=-2)

    h = torch.fft.fftn(y)[:, :, :, None] * torch.fft.fftn(v)
    out = torch.fft.ifft(h)[:, :, : x.size(-2), :]
    return out


# def fft_multiply(A, x):
#     n = A.shape[0]
#     idx = [i for i in range(A[0].size(0) - 1, -1, -1)]
#     idx = torch.LongTensor(idx)
#     inverted_tensor_1 = A[0].index_select(0, idx)

#     idx = [i for i in range(A[0].size(0) - 1, 0, -1)]
#     idx = torch.LongTensor(idx)
#     inverted_tensor_2 = A.index_select(0, idx)

#     y = torch.cat([inverted_tensor_1, inverted_tensor_2[0]])

#     zeros = torch.zeros((x.size(0), y.size(0) - x.size(1))).float()
#     v = torch.cat((x, zeros), dim=1)

#     h = torch.fft.fft(y) * torch.fft.fftn(v)
#     out = torch.fft.ifft(h)[:, : x.shape[1]]
#     return out
