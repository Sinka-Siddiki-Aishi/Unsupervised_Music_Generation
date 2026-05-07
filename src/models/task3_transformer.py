import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class Task3Transformer(nn.Module):
    def __init__(
        self,
        input_size=128,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ):
        super(Task3Transformer, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_model, input_size)
        self.sigmoid = nn.Sigmoid()

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x):
        """
        x shape: batch_size x seq_len x 128
        """
        seq_len = x.size(1)
        device = x.device

        x = self.input_projection(x)
        x = self.positional_encoding(x)

        causal_mask = self.generate_causal_mask(seq_len, device)

        x = self.transformer(x, mask=causal_mask)

        output = self.output_layer(x)
        output = self.sigmoid(output)

        return output


