import torch
import torch.nn as nn
import numpy as np


class SingleHeadAttention(nn.Module):
    """Single Attention Head
    
    Arguments
    ---------
    Q : torch.tensor [bs, seq_len, d_model]
    K : torch.tensor [bs, seq_len, d_model]
    V : torch.tensor [bs, seq_len, d_model]
    
    Returns
    -------
    Y : torch.tensor [bs, seq_len, d_v]
    """

    def __init__(self, d_model=8, d_q=8, d_k=8, d_v=8, max_seq_len=10):
        super(SingleHeadAttention, self).__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.query_weights = nn.ModuleList(
            [nn.Linear(d_model, d_q) for _ in range(max_seq_len)]
        )

        self.key_weights = nn.ModuleList(
            [nn.Linear(d_model, d_k) for _ in range(max_seq_len)]
        )

        self.value_weights = nn.ModuleList(
            [nn.Linear(d_model, d_v) for _ in range(max_seq_len)]
        )

    def linear(self, Q, K, V):

        Q = torch.stack(
            [self.query_weights[ts](Q[:, ts, :]) for ts in range(Q.shape[1])],
            dim=1,
        )

        K = torch.stack(
            [self.key_weights[ts](K[:, ts, :]) for ts in range(K.shape[1])],
            dim=1,
        )

        V = torch.stack(
            [self.value_weights[ts](V[:, ts, :]) for ts in range(V.shape[1])],
            dim=1,
        )
        return Q, K, V

    def attention(self, Q, K, V):
        S = torch.bmm(Q, K.permute(0, 2, 1)) / torch.sqrt(
            torch.tensor(self.d_k)
        )
        W = nn.Softmax(dim=2)(S)
        Y = torch.bmm(W, V)
        return Y

    def forward(self, Q, K, V):
        Q, K, V = self.linear(Q, K, V)
        Y = self.attention(Q, K, V)
        return Y


class MaskedSingleHeadAttention(SingleHeadAttention):
    """Single Attention Head
    
    Arguments
    ---------
    Q : torch.tensor [bs, seq_len, d_model]
    K : torch.tensor [bs, seq_len, d_model]
    V : torch.tensor [bs, seq_len, d_model]
    
    Returns
    -------
    Y : torch.tensor [bs, seq_len, d_v]
    
    
    Difference Compared to Regular SingleHeadAttention is the changed Attention weight Matrix.
    It ensures that every Prediction timestep has only access to itself and previous timesteps.
    
    [w 0 0 0 0]    |
    [w w w 0 0]   output
    [w w w w 0]    |
    [w w w w w]    |
    
    <--input-->
     
    """

    def __init__(self):
        super(MaskedSingleHeadAttention, self).__init__()

    def put_on_mask(self, S):

        actual_seq_len = S.shape[1]

        mask = torch.ones((actual_seq_len, actual_seq_len))
        neg_infs = torch.zeros((actual_seq_len, actual_seq_len))

        for t in range(actual_seq_len):
            for d in range(actual_seq_len):
                if d > t:
                    mask[t, d] = 0
                    neg_infs[t, d] = -np.inf

        S_masked = S * mask + neg_infs
        return S_masked

    def attention(self, Q, K, V):
        S = torch.bmm(Q, K.permute(0, 2, 1)) / torch.sqrt(
            torch.tensor(self.d_k)
        )
        S = self.put_on_mask(S)
        W = nn.Softmax(dim=2)(S)
        Y = torch.bmm(W, V)
        return Y


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_v, d_model):
        super(MultiHeadAttention, self).__init__()

        self.h = h

        self.heads = nn.ModuleList([SingleHeadAttention() for _ in range(h)])

        self.dense = nn.Linear(h * d_v, d_model)

    def forward(self, Q, K, V):
        y_SH_cat = torch.cat(
            [self.heads[i](Q, K, V) for i in range(self.h)], dim=2
        )
        y_MH = self.dense(y_SH_cat)
        y_MH = nn.Softmax(dim=2)(y_MH)
        return y_MH


class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, h):
        super(MaskedMultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList(
            [MaskedSingleHeadAttention() for _ in range(h)]
        )

