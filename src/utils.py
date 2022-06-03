import torch
import math


def batch_to_one_hot(batch_cat_id, num_cats):
    """
    Arguments
    ---------
    batch_cat_id : torch.tensor [bs, seq_len, 1]
    
    Returns
    -------
    batch_cat_OH : torch.tensor [bs, seq_len, num_cats]
    
    """
    cat_samples = batch_cat_id.chunk(len(batch_cat_id), dim=0)
    batch_cat_OH = list()
    for cat_sample in cat_samples:
        cat_id = cat_sample.squeeze()
        cat_OH = torch.zeros(len(cat_id), num_cats)
        cat_OH[torch.arange(len(cat_id)), cat_id] = 1
        batch_cat_OH.append(cat_OH)

    return torch.stack(batch_cat_OH, dim=0)


def positional_encoding(seq_len, d_model):
    def p(t, k):
        def is_even(x):
            return x % 2 == 0

        def w(k):
            return torch.tensor(1 / math.pow(10000, 2 * k / d_model))

        if is_even(k):
            return torch.sin(w(k) * t)
        if not is_even(k):
            return torch.cos(w(k) * t)

    P = torch.zeros((seq_len, d_model))
    for t in range(seq_len):
        for k in range(d_model):
            P[t, k] = p(t, k)
    return P


def binary_to_float(x):
    d = len(x)
    return torch.tensor([math.pow(2, idx) * x[idx] for idx in range(d)]).sum()
