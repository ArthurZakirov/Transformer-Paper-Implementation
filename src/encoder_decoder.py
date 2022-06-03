import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.attention import MaskedMultiHeadAttention


class EncoderLayer(nn.Module):
    """Single Encoder Layer 
    
    Attributes
    ----------
    multi_head_attention
    add_and_norm
    
    feed_forward 
    add_and_norm
    
    """

    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention()
        self.add_and_norm_MHA = AddAndNorm()

        self.feed_forward = FeedForward()
        self.add_and_norm_ff = AddAndNorm()

    def forward(self, x_emb):
        MHA = self.multi_head_attention(Q=x_emb, K=x_emb, V=x_emb)
        MHA_norm = self.add_and_norm_MHA(MHA, x_emb)

        ff = self.feed_forward(MHA_norm)
        z = self.add_and_norm_ff(ff, MHA_norm)

        return z


class DecoderLayer(nn.Module):
    """Single Decoder Layer 
    
    Attributes
    ----------
    masked_multi_head_attention 
    add_and_norm
    
    multi_head_attention
    add_and_norm
    
    feed_forward 
    add_and_norm
    
    """

    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.masked_multi_head_attention = MaskedMultiHeadAttention()
        self.add_and_norm_MMHA = AddAndNorm()

        self.multi_head_attention = MultiHeadAttention()
        self.add_and_norm_MHA = AddAndNorm()

        self.feed_forward = FeedForward()
        self.add_and_norm_ff = AddAndNorm()

    def forward(self, output, latent):
        MMHA = self.masked_multi_head_attention(Q=output, K=output, V=output)
        MMHA_norm = self.add_and_norm_MMHA(output, MMHA)

        MHA = self.multi_head_attention(Q=MMHA_norm, K=latent, V=latent)
        MHA_norm = self.add_and_norm_MHA(MMHA_norm, MHA)

        ff = self.feed_forward(MHA_norm)
        ff_norm = self.add_and_norm_ff(ff, MHA_norm)

        return ff_norm


class AddAndNorm(nn.Module):
    """Add State and it's copy that has passed through Attention and use Layer Normalization
    """

    def __init__(self, d_model):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(d_model)

    def forward(self, x, y):
        return self.normalize(x + y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()

        self.linear_in = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = nn.ReLU()(self.linear_in(x))
        y = self.linear_out(h)
        return y


class Encoder(nn.Module):
    """Sequence of Encoder Layers
    
    Attributes
    ----------
    N : int
        Number Encoder Layers
        
    """

    def __init__(self, N):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N)])

    def forward(self, x_emb):
        z = x_emb
        N_z = list()
        for layer in self.layers:
            z = layer(z)
            N_z.append(z)

        return N_z


class Decoder(nn.Module):

    """Sequence of Decoder Layers
    
    Attributes
    ----------
    N : int
        Number Decoder Layers
        
    """

    def __init__(self, N):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N)])

    def forward(self, y, N_z):
        for n, layer in enumerate(self.layers):
            z = N_z[n]
            y = layer(y, z)
        return y
