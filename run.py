from attention import RPE_NKA
import scipy.linalg
import numpy as np

import torch

if __name__ == "__main__":

    attn = RPE_NKA(dim=512, max_len=1024)
    x = torch.randn(1, 1024, 512)

    out = attn(x)
