import torch
import torch.nn as nn


class Task1LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, latent_size=64, num_layers=1):
        super(Task1LSTMAutoencoder, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        last_hidden = hidden[-1]
        z = self.to_latent(last_hidden)
        return z

    def decode(self, z, seq_len):
        hidden = self.from_latent(z)
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder_lstm(decoder_input)
        output = self.output_layer(decoded)
        output = self.sigmoid(output)

        return output

    def forward(self, x):
        seq_len = x.shape[1]
        z = self.encode(x)
        reconstructed = self.decode(z, seq_len)
        return reconstructed


