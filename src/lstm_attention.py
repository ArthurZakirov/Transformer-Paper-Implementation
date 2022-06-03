import torch
import torch.nn as nn


bs = 256
input_dim = 10
hidden_dim = 32
output_dim = 10
ht = 8
ft = 3


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input_embedder = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, x):
        enc_input = self.input_embedder(x)
        out, (h, c) = self.encoder(enc_input)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

        self.MLP_q = nn.Linear(hidden_dim, hidden_dim)
        self.MLP_k = nn.Linear(hidden_dim, hidden_dim)
        self.MLP_v = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, h_enc, h_dec):
        q = self.MLP_q(h_dec).unsqueeze(1)

        score_list = list()
        for t in range(ht):
            k = self.MLP_k(h_enc[:, t, :]).unsqueeze(1)

            q_dot_k = torch.bmm(q, k.permute(0, 2, 1)).squeeze(2).squeeze(1)
            sqrt_H = torch.tensor(hidden_dim).float().sqrt()
            score = q_dot_k / sqrt_H
            score_list.append(score)

        score = torch.stack(score_list, dim=1)
        att_q_k = nn.Softmax(dim=1)(score).unsqueeze(2)
        return att_q_k

    def context(self, att, h_enc):
        v = self.MLP_v(h_enc)
        context = (att * v).sum(dim=1)
        return context

    def forward(self, h_enc, h_dec):
        attention = self.attention(h_enc, h_dec)
        context = self.context(attention, h_enc)
        return context


class AdditiveAttention(nn.Module):
    def __init__(self):
        super(AdditiveAttention, self).__init__()

        self.Wq_dot_ = nn.Linear(hidden_dim, hidden_dim)
        self.Wk_dot_ = nn.Linear(hidden_dim, hidden_dim)
        self.wa_dot_ = nn.Linear(hidden_dim, 1)

    def forward(self, h_enc, h_dec):

        # Calculate Attention
        q = h_dec
        score_list = list()
        for t in range(ht):
            k = h_enc[:, t, :]
            Wk_dot_k = self.Wk_dot_(k)

            q = h_dec.squeeze()
            Wq_dot_q = self.Wq_dot_(q)

            a = nn.Tanh()(Wq_dot_q + Wk_dot_k)
            wa_dot_a = self.wa_dot_(a)

            score_list.append(wa_dot_a)
        score = torch.stack(score_list, dim=1)

        att_q_k = nn.Softmax(dim=1)(score)

        # Calculate Context Vector
        v = h_enc
        context = (att_q_k * v).sum(dim=1)

        return context


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        # self.attention = AdditiveAttention()
        self.attention = ScaledDotProductAttention()
        self.output_embedder = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_enc):

        h = torch.zeros(h_enc.shape[0], hidden_dim)
        c = torch.zeros(h_enc.shape[0], hidden_dim)

        h_dec = list()
        for t in range(ft):
            x = self.attention(h_enc, h)
            h, c = self.decoder_cell(x, (h, c))
            h_dec.append(h)

        h_dec = torch.stack(h_dec, dim=1)

        logits = self.output_embedder(h_dec)
        log_pis = nn.LogSoftmax(dim=-1)(logits)
        return log_pis


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):

        h_enc = self.encoder(x)
        y = self.decoder(h_enc)
        return y


def CrossEntropyLoss(log_p_y_x, y):

    y_OH = batch_to_one_hot(y, num_cats)
    E_i_t = (log_p_y_x * y_OH).sum(dim=2)
    E_i = E_i_t.sum(dim=1)
    E = E_i.mean(dim=0)

    return -E
