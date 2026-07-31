import torch

def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k**0.5
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

Q = torch.randn(6, 8)
K = torch.randn(6, 8)
V = torch.randn(6, 8)
print(attention(Q, K, V).shape)
