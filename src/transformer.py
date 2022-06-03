import torch
import torch.nn as nn

from src.encoder_decoder import Encoder, Decoder
from src.utils import positional_encoding, batch_to_one_hot


class Transformer(nn.Module):
    """Transformer - Timeseries Forecasting
    
    Attributes
    ----------
    task_type : str 
    
        'regression' : input / output OHE [bs, time, cat]   
        'classification' : input / output [bs, time, dim]
    
    Methods
    -------
    run_encoder()
        Run all parts of the Encoding process
        
    run_decoder_inference()
        Run all parts of the Decoding process in the Prediction Mode
        
    run_decoder_train()
        Run all parts of the Decoding process in the Training Mode, when labels are known in advance
        
    predict()
        Run full Trainsformer in the Prediction Mode, returning a prediction
        
    train_loss()
        Run full Trainsformer in the Training Mode, returning the training loss
        
    """

    def __init__(self, d_input, d_model, d_output, seq_len, ft, d):
        super(Transformer, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.seq_len = seq_len
        self.ft = ft
        self.d = d

        self.task_type = "regression"  # 'classification'

        self.enc_input_embedder = nn.Linear(d_input, d_model)
        self.encoder = Encoder()

        self.dec_ouput_embedder = nn.Linear(d_input, d_model)
        self.decoder = Decoder()

        self.dec_linear = nn.Linear(d_model, d_output)

    def run_encoder(self, x):
        """Run All parts of the Encoding process
        
        Arguments
        ---------
        x : torch.tensor [bs, ht, d_input]
        
        Returns
        -------
        N_z : list of torch.tensor [bs, ht, d_model]
        
        """

        x_emb = self.enc_input_embedder(x)
        x_PE = positional_encoding(self.seq_len, self.d_model)
        x_enc_in = x_emb + x_PE

        N_z = self.encoder(x_enc_in)
        return N_z

    def run_decoder_inference(self, y_previous, N_z):
        """ Decoder Inference Mode: Iteratively
        
        Arguments
        ---------
        N_z : list of torch.tensor [bs, ht, d_model]
            Encoder outputs of history sequence of each of the N layers.
            
        y_previous : torch.tensor [bs, 1, d_input] 
            Initial decoder input. 
            We just copy the last timestep of the history sequence here.
            
        Procedure
        ---------
        First input to the decoder is just like start token.
        The last output timestep of the decoder gets apppended to the input for the next iteration.
        With every Iteration, both the in and output sequence grow, until the desired sequence length is reached.
        The last output sequence is the final output.
        
        Iteration 1: in:[y0]       > out:[y1] 
        Iteration 2: in:[y0,y1]    > out:[y1,y2]
        Iteration 3: in:[y0,y1,y2] > out:[y1,y2,y3]
        
        y_pred = [y1, y2, ..., y_ft]
        
        Returns
        -------
        y_pred : torch.tensor [bs, ft, d_output]
        
        """
        for t in range(self.ft):

            # embedding
            y_emb = self.dec_ouput_embedder(y_previous)
            y_PE = positional_encoding(y_emb.shape[1], self.d_model)
            y_dec_in = y_emb + y_PE

            # decoder
            y_next = self.decoder(y_dec_in, N_z)
            y_next = y_next[:, -1, :].unsqueeze(1)
            y_next = self.dec_linear(y_next)

            # Ouput Distribution
            if self.task_type == "classification":
                p_y_x = nn.Softmax(dim=-1)(y_next)
                y_next = p_y_x

            if self.task_type == "regression":
                # hier gegebenfalls Normal distribution aufstellen
                mu = y_next
                y_next = mu

            y_previous = torch.cat([y_previous, y_next], dim=1)

        y_pred = y_previous[:, 1:, :]

        if self.task_type == "classification":
            cat_id = y_pred.argmax(dim=-1).unsqueeze(-1)
            cat_OH = batch_to_one_hot(cat_id, self.d)
            return cat_OH

        if self.task_type == "regression":
            return y_pred

    def run_decoder_train(self, y_0, y_gt, N_z):

        """ Decoder Training Mode: Simultaniously all timesteps
        
        Arguments
        ---------
        N_z : list of torch.tensor [bs, ht, d_model]
            Encoder outputs of history sequence of each of the N layers.
            
        y_0 : torch.tensor [bs, 1, d_input] 
            Initial decoder input. 
            We just copy the last timestep of the history sequence here.
            
        y_gt : torch.tensor [bs, ft, d_output]
            
        Procedure
        ---------
        All timesteps are calculated simultaneously.
        
        
        in: [y0, y1, ..., y_ft-1] -> out: [y1, y2, ... , y_ft]
        
          cat[y0 | y_gt[:-1]]     ->           y_gt
                
        Returns
        -------
        y_pred : torch.tensor [bs, ft, d_output]
        
        """
        # concat first token to ground truth
        y_shifted_right = torch.cat([y_0, y_gt[:, :-1, :]], dim=1)

        # embedder
        y_emb = self.dec_ouput_embedder(y_shifted_right)
        y_PE = positional_encoding(y_emb.shape[1], self.d_model)
        y_dec_in = y_emb + y_PE

        # decoder
        y_pred = self.decoder(y_dec_in, N_z)
        y_pred = self.dec_linear(y_pred)

        # Ouput Distribution + Loss
        if self.task_type == "classification":
            p_y_x = nn.Softmax(dim=-1)(y_pred)
            loss = CESequenceLoss(p_y_x, y_gt)

        if self.task_type == "regression":
            # Hier gegebnfalls Normal Distribution
            mu = y_pred
            y_pred = mu
            loss = MSESequenceLoss(y_pred, y_gt)

        return loss

    def predict(self, x):
        N_z = self.run_encoder(x)
        first_dec_out = x[:, -1, :].unsqueeze(1)
        y_pred = self.run_decoder_inference(first_dec_out, N_z)
        return y_pred

    def train_loss(self, x, y_gt):
        N_z = self.run_encoder(x)
        y_0 = x[:, -1, :].unsqueeze(1)
        loss = self.run_decoder_train(y_0, y_gt, N_z)
        return loss


def CESequenceLoss(p_y_x, y):
    log_p_y_x = torch.log(p_y_x).clamp(min=-100)
    E_i_t = -(y * log_p_y_x).sum(dim=2)
    E_i = E_i_t.sum(dim=1)
    E = E_i.mean(dim=0)
    return E


def MSESequenceLoss(y_pred, y):
    SE_dim_total = ((y_pred - y) ** 2).sum(2)
    SE_seq_total = SE_dim_total.sum(1)
    MSE = SE_seq_total.mean(dim=0)
    return MSE
