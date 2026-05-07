import torch
import torch.nn as nn


class Task2VAE(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, latent_size=64, num_layers=1):
        super(Task2VAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

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

        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
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

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, seq_len)

        return reconstructed, mu, logvar


