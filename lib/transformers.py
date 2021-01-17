import torch
import torch.nn as nn
import math
from torch import optim


# The widely used Positional Encoding implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmbeddedTransformer(nn.Module):
    def __init__(self,
                 d_model,
                 src_vocab,
                 trg_vocab,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 device,
                 pad_idx,
                 max_len=100):
        super(EmbeddedTransformer, self).__init__()

        # Params
        self.d_model = d_model
        self.device = device
        self.pad_idx = pad_idx

        # Model
        self.embed_src = nn.Embedding(src_vocab, d_model)
        self.embed_trg = nn.Embedding(trg_vocab, d_model)
        self.embed_src_pos = PositionalEncoding(d_model, dropout, max_len)
        self.embed_trg_pos = PositionalEncoding(d_model, dropout, max_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, trg_vocab)

        # Initialize parameters
        # Warning: no initialization is mentioned in the original paper
        # To follow Attention Is All You Need, comment out the following line:
        self.init_params()

    def init_params(self):
        # As noted in several other sources (not the original paper),
        # Xavier initialization drastically improves model performance

        for params in self.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

    def forward(self, src, trg):

        # Unembedded data
        # src: [S, N]
        # trg: [T, N]

        # First, prepare masks
        src_key_padding_mask = (src.transpose(0, 1) == self.pad_idx).to(self.device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(self.device)

        # src_key_padding_mask: [N, S]
        # trg_mask: [T, T]

        # Embed and encode
        # src_pos: [S, N]
        # trg_pos: [T, N]

        src = self.embed_src(src) * math.sqrt(self.d_model)
        src = self.embed_src_pos(src)
        src = self.dropout(src)

        trg = self.embed_trg(trg) * math.sqrt(self.d_model)
        trg = self.embed_trg_pos(trg)
        trg = self.dropout(trg)

        # Embedded data
        # src: [S, N, E]
        # trg: [T, N, E]

        out = self.transformer(src, trg, src_key_padding_mask=src_key_padding_mask, tgt_mask=trg_mask)

        # out: [T, N, E]

        out = self.fc(out)

        # V = len(TRG_VOCAB)
        # out: [T, N, V]

        return out


class OptimWrapper:

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.step_num += 1
        new_lr = self.get_lr(self.step_num)

        # Update wrapped optimizer learning rate
        for p in self.optimizer.param_groups:
            p['lr'] = new_lr

        self.optimizer.step()

    def get_lr(self, step):
        return (self.d_model ** (-0.5)) * min(step ** (-0.5),
                                              step * (self.warmup_steps ** (-1.5)))


def build_transformer(
        src_vocab,
        trg_vocab,
        src_pad_idx,
        trg_pad_idx,
        device,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=100,
        warmup_steps=4000):
    # Model
    model = EmbeddedTransformer(d_model, src_vocab, trg_vocab, nhead,
                                num_encoder_layers, num_decoder_layers,
                                dim_feedforward, dropout,
                                device, src_pad_idx, max_len)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98))
    optimizer = OptimWrapper(optimizer, d_model, warmup_steps)

    # Criterion
    # Possible addition - a label smoothing module
    # Didn't manage to make one myself
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    return model, optimizer, criterion
